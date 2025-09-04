from torch import nn
import torch
from timm.models.layers import trunc_normal_, DropPath, to_2tuple
from functools import partial

class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None):
        super(CrossAttention, self).__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.kv1 = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.kv2 = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.down = nn.Conv2d(dim * 2, dim, kernel_size=1)

    def forward(self, x1, x2):
        # B, C, W, H = x1.shape
        B, N, C = x1.shape
        # x1 = x1.reshape(B, H * W, C)
        # x2 = x2.reshape(B, H * W, C)
        q1 = x1.reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        q2 = x2.reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        k1, v1 = self.kv1(x1).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        k2, v2 = self.kv2(x2).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()

        ctx1 = (k1.transpose(-2, -1) @ v1) * self.scale
        ctx1 = ctx1.softmax(dim=-2)
        ctx2 = (k2.transpose(-2, -1) @ v2) * self.scale
        ctx2 = ctx2.softmax(dim=-2)

        # x1 = (q1 @ ctx2).permute(0, 2, 1, 3).reshape(B, W * H, C).contiguous() 
        # x2 = (q2 @ ctx1).permute(0, 2, 1, 3).reshape(B, W * H, C).contiguous() 
        x1 = (q1 @ ctx2).permute(0, 2, 1, 3).reshape(B, N, C).contiguous() 
        x2 = (q2 @ ctx1).permute(0, 2, 1, 3).reshape(B, N, C).contiguous() 
        # x = x1 + x2
        # x = torch.concat((x1, x2), dim=2)
        # x = x.unsqueeze(0).reshape(B, C * 2, H, W).contiguous()
        # x = self.down(x)
        # x = x.permute(0, 2, 1, 3).reshape(B, W * H, C).contiguous()
        return x1, x2
        # return x1, x2

class Cross(nn.Module):
    def __init__(self, dim, reduction=1, num_heads=None, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.channel_proj1 = nn.Linear(dim, dim // reduction * 2)
        self.channel_proj2 = nn.Linear(dim, dim // reduction * 2)
        self.act1 = nn.ReLU(inplace=True)
        self.act2 = nn.ReLU(inplace=True)
        self.cross_attn = CrossAttention(dim // reduction, num_heads=num_heads)
        # self.end_proj1 = nn.Linear(dim // reduction * 2, dim)
        # self.end_proj2 = nn.Linear(dim // reduction * 2, dim)
        self.end_proj = nn.Linear(dim // reduction * 2, dim)
        # self.norm1 = nn.LayerNorm(dim)
        # self.norm2 = nn.LayerNorm(dim)
        self.norm = nn.LayerNorm(dim)
        self.channel = ChannelEmbed(in_channels=dim, out_channels=dim, reduction=reduction, norm_layer=norm_layer)

    def forward(self, x1, x2):
        B, C, H, W = x1.shape
        x1 = x1.flatten(2).transpose(1, 2)
        x2 = x2.flatten(2).transpose(1, 2)
        y1, u1 = self.act1(self.channel_proj1(x1)).chunk(2, dim=-1)
        y2, u2 = self.act2(self.channel_proj2(x2)).chunk(2, dim=-1)
        v1, v2 = self.cross_attn(u1, u2)
        # fusion = torch.concat((v1, v2), dim=-1)
        # ori_fusion = torch.concat((y1, y2), dim=-1)
        # fusion = fusion + ori_fusion
        fusion_rgb = torch.concat((v1, y1), dim=-1)
        fusion_the = torch.concat((v2, y2), dim=-1)
        fusion = fusion_rgb + fusion_the
        fusion = self.end_proj(fusion)
        out1 = self.norm(fusion)
        merge = out1 + x1 + x2
        
        # y1 = torch.cat((y1, v1), dim=-1)
        # y2 = torch.cat((y2, v2), dim=-1)
        # out_x1 = self.norm1(x1 + self.end_proj1(y1))
        # out_x2 = self.norm2(x2 + self.end_proj2(y2))
        # out = torch.concat((out_x1, out_x2), dim=-1)
        out = self.channel(merge, H, W)
        out = out.permute(0, 2, 1, 3).reshape(B, H * W, C)
        return out
    
class ChannelEmbed(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=1, norm_layer=nn.BatchNorm2d):
        super(ChannelEmbed, self).__init__()
        self.out_channels = out_channels
        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.channel_embed = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels//reduction, kernel_size=1, bias=True),
                        nn.Conv2d(out_channels//reduction, out_channels//reduction, kernel_size=3, stride=1, padding=1, bias=True, groups=out_channels//reduction),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(out_channels//reduction, out_channels, kernel_size=1, bias=True),
                        norm_layer(out_channels) 
                        )
        self.norm = norm_layer(out_channels)
        
    def forward(self, x, H, W):
        B, N, _C = x.shape
        x = x.permute(0, 2, 1).reshape(B, _C, H, W).contiguous()
        residual = self.residual(x)
        x = self.channel_embed(x)
        out = self.norm(residual + x)
        return out

# # 分割窗口
# def window_partition(x, window_size):
#     """
#     x:(B, H, W, C)
#     window_size:int
#     returns:
#             windows:(num_windows*B, window_size, window_size, C)
#     """
#     # print(x.shape)
#     B, H, W, C = x.shape
#     # print(H / window_size)
#     x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
#     windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
#     return windows

# #将分割后的窗口重新组成原来图像的形状
# def window_reverse(windows, window_size, H, W):
#     B = int(windows.shape[0] / (H * W / window_size / window_size))
#     x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
#     x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
#     return x

# # W-MSA: 注意力中加入了位置偏移，在这个模块中要计算相对位置偏移表和对应的索引
# class WindowAttention(nn.Module):
#     def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
#         """
#         Args:
#             dim: input channels(int)
#             window_size: height and weight of the window(tuple[int])
#             num_heads: number pf attention heads
#             qkv_bias: if true, add a learnable bias to query, key, value(bool, optional)
#             attn_drop: Dropout ratio of attention weight(float, optional)
#             proj_drop: Dropout ratio of output(float, optional)
#         """
#         super().__init__()
#         self.dim = dim
#         self.window_size = window_size #Wh, Ww
#         self.num_heads = num_heads
#         head_dim = dim // num_heads
#         self.scale = head_dim ** -0.5
#         # define a parameter table of relative position bias
#         self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * window_size[0] - 1)* (2 * window_size[1] - 1), num_heads)) # 2*Wh-1 * 2*Ww-1, nH
#         # get pair-wise relative position index for each token inside the window
#         coords_h = torch.arange(self.window_size[0])
#         coords_w = torch.arange(self.window_size[1])
#         # torch.meshgrid():创建一个坐标网格,返回两个二维张量，分别表示网格中每个点的行坐标和列坐标
#         coords = torch.stack(torch.meshgrid([coords_h, coords_w])) # 2, Wh, Ww
#         coords_flatten = torch.flatten(coords, 1) # 2, Wh*Ww
#         # 计算相对坐标，虽然维度完全一样但不能直接进行相减运算，需要进行广播操作，否则会得到所有元素都为0的张量
#         relative_coords = coords_flatten[:, :, None] - coords_flatten[: ,None ,:] # 2, Wh*Ww, Wh*Ww
#         relative_coords = relative_coords.permute(1, 2, 0).contiguous() # Wh*Ww, Wh*Ww, 2
#         # 确保每个位置上的相对位置索引是唯一的
#         relative_coords[:, :, 0] += self.window_size[0] - 1
#         relative_coords[:, :, 1] += self.window_size[1] - 1
#         relative_coords[:, :, 0] *= 2 * self.window_size[0] - 1
#         relative_position_index = relative_coords.sum(-1) # Wh*Ww, Wh*Ww
#         # self.register_buffer:注册一个不会参与模型参数梯度计算的张量(buffer)
#         self.register_buffer("relative_position_index", relative_position_index)
#         self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(proj_drop)
#         # 用来自截断正态分布的值进行填充，排除了距离均值超过两个标准差之外的值，从而避免了极端值的出现
#         trunc_normal_(self.relative_position_bias_table, std=.02)
#         self.softmax = nn.Softmax(dim=-1)

#     def forward(self, x, mask=None):
#         B_, N, C = x.shape
#         qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
#         q, k, v = qkv[0], qkv[1], qkv[2]
#         q = q * self.scale
#         attn = (q @ k.transpose(-2, -1))
#         relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1) #Wh*Ww, Wh*Ww, nH
#         relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous() # nH, Wh*Ww, Wh*Ww
#         attn = attn + relative_position_bias
#         # 用于窗口融合时屏蔽来自不同窗口的信息
#         if mask is not None:
#             nW = mask.shape[0]
#             # B_ // nW:调整后的批次大小(即应用掩码后，每个子批次中剩余的样本数量)，nW是应用掩码后每个子批次中的样本数量.
#             attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
#             attn = attn.view(-1, self.num_heads, N, N)
#             attn = self.softmax(attn)
#         else:
#             attn = self.softmax(attn)
#         attn = self.attn_drop(attn)
#         x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
#         x = self.proj(x)
#         x = self.proj_drop(x)
#         return x

# class Window(nn.Module):
#     def __init__(self, dim, input_resolution, num_heads, window_size=5, shift_size=0, qkv_bias=True, drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, fused_window_process=False):
#         super().__init__()
#         self.dim = dim 
#         # self.num_head_cross = num_heads_cross
#         self.input_resolution = input_resolution
#         self.num_head_window = num_heads
#         self.window_size = window_size
#         self.shift_size = shift_size
#         # a = min(self.input_resolution)
#         # print(a)
#         # print(self.input_resolution)
#         # print(type(self.input_resolution))
#         if min(self.input_resolution) <= self.window_size:
#             self.shift_size = 0
#             self.window_size = min(self.input_resolution)
#         assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"
#         # self.attn_cross = CrossAttention(dim, num_heads=num_heads_cross)
#         self.norm1 = norm_layer(dim)
#         # 灵活的加入DropPath正则化
#         self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
#         self.attn_window = WindowAttention(dim, window_size=to_2tuple(self.window_size), num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
#         if self.shift_size > 0:
#             # calculate attention mask for SW-MSA
#             H, W = self.input_resolution
#             img_mask = torch.zeros((1, H, W, 1))
#             # slice(0, -self.window_size): 从图像的起始位置到倒数self.window_size个元素之前的范围。通常用于获取不包括边界的中心区域
#             # slice(-self.window_size, -self.shift_size): 从倒数self.window_size个元素到倒数self.shift_size个元素之前的范围。通常获取靠近到没有到达实际边界的区域
#             # slice(-self.shift_size, None):从倒数self.shift_size个元素到图像的末尾。None表示切片的结束位置没有限制，即一直延伸到图像的最后
#             h_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
#             w_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
#             cnt = 0
#             # 为每个区域赋予一个值，不同的区域cnt值不同。在注意力计算时用于指示哪些区域是有效的
#             for h in h_slices:
#                 for w in w_slices:
#                     img_mask[:, h, w, :] = cnt
#                     cnt += 1
#             mask_windows = window_partition(img_mask, self.window_size) #nW*B, window_size, window_size, 1
#             # 将每个窗口展平为向量
#             mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
#             # 计算注意力掩码
#             attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
#             # 第一个masked_fill：将所有非0元素替换成一个负数（经过softmax后值为0，达到屏蔽不是同一个窗口的信息的目的）
#             # 第二个masked_fill：将所有0元素替换为0，保留了这些位置的原始权重
#             attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
#         else:
#             attn_mask = None
#         self.register_buffer("attn_mask", attn_mask)
#         self.fused_window_process = fused_window_process
    
#     def forward(self, x):
#         # print(type(x))
#         shortcut = x
#         # x = self.attn_cross(x)
#         H, W = self.input_resolution
#         # print(x.shape)
#         B, L, C = x.shape
#         x = self.norm1(x)
#         x = x.view(B, H, W, C)
#         # cyclic shift
#         if self.shift_size > 0:
#             # 滚动窗口 向右下方移位
#             shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
#             # partition windows
#             x_windows = window_partition(shifted_x, self.window_size) #nW*B, window_size, window_size, C
#         else:
#             shifted_x = x
#             # partition windows
#             x_windows = window_partition(shifted_x, self.window_size) #nW*B, window_size, window_size, C
#         x_windows = x_windows.view(-1, self.window_size * self.window_size, C) #nW*B, window_size*window_size, C
#         #W-MSA/SW-MSA
#         attn_windows = self.attn_window(x_windows, mask=self.attn_mask) #nW*B, window_size*window_size, C
#         # merge windows
#         attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
#         # reverse cyclic shift
#         if self.shift_size > 0:
#             shifted_x = window_reverse(attn_windows, self.window_size, H, W) #B, H, W, C
#             # print(shifted_x.shape)
#             # 向左上方移位
#             x = torch.roll(input=shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
#         else:
#             shifted_x = window_reverse(attn_windows, self.window_size, H, W)
#             x = shifted_x
#         x = x.view(B, H * W, C)
#         x = shortcut + self.drop_path(x)
#         return x

class CrossWindow(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.dim = dim
        # self.input_resolution = input
        # build blocks
        self.cross = Cross(dim=dim, reduction=1, num_heads=num_heads)
        # self.spatial_attn = SpatialAttention()

        # self.blocks = nn.ModuleList([Window(dim=dim, num_heads=num_heads, input_resolution=input, window_size=window_size, 
        #                                     shift_size=0 if i % 2 != 0 else window_size // 2, qkv_bias=qkv_bias, drop=drop, attn_drop=attn_drop,
        #                                          drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path, norm_layer=norm_layer, fused_window_process=fused_window_process)
        #                                          for i in range(1, depth + 1)])
        
    def forward(self, rgb_x, the_x):
        values = self.cross(rgb_x, the_x)
        # return values
        # x = self.spatial_attn(values)
        # attn_weights, values = self.cross(rgb_x, the_x)
        # print(type(values))
        # for blk in self.blocks:
        #     x = blk(values)
        return values