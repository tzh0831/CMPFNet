from torch import nn
import torch
from timm.models.layers import trunc_normal_

class Weighted(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.left = nn.Sequential(
            nn.Conv2d(2 * dim, 2 * dim, kernel_size=1, bias=True),
            nn.Conv2d(2 * dim, dim, kernel_size=3, stride=1, padding=1, bias=True),
            nn.Sigmoid()
        )
        self.right = nn.Sequential(
            nn.Conv2d(2 * dim, 2 * dim, kernel_size=1, bias=True),
            nn.Conv2d(2 * dim, dim, kernel_size=3, stride=1, padding=1, bias=True),
            nn.Tanh()
        )
    
    def forward(self, x1, x2):
        # x1为RGB图像 x2为红外图像
        B, _, H, W = x1.shape
        x = torch.cat((x1, x2), dim=1)
        x_left = self.left(x)
        x_right = self.right(x)
        x = x_left * x_right
        avg = self.avg(x).view(B, self.dim)
        avg = avg.reshape(B, self.dim, 1, 1)
        avg_right = 1 - avg
        avg_right = avg_right.reshape(B, self.dim, 1, 1)
        x_left = avg * x1
        x_right = avg_right * x2
        return x_left, x_right

# Feature Rectify Module
class ChannelRect(nn.Module):
    def __init__(self, dim, reduction=1) -> None:
        super().__init__()
        self.dim = dim
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.max = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(self.dim*4, self.dim*4 // reduction),
            nn.ReLU(inplace=False),
            nn.Linear(self.dim*4 // reduction, self.dim*2),
            nn.Sigmoid()
            # nn.ReLU()
        )
    
    def forward(self, rgb_x, the_x):
        B, _, H, W = rgb_x.shape
        # 对照着框架图看一下每层通道数的尺寸
        # print(rgb_x.shape)
        # print(self.avg(rgb_x).shape)
        rgb_avg_x = self.avg(rgb_x).view(B, self.dim)
        # print(rgb_avg_x.shape)
        rgb_max_x = self.max(rgb_x).view(B, self.dim)
        # print(rgb_max_x.shape)
        rgb = torch.concat((rgb_avg_x, rgb_max_x), dim=1)
        # print(rgb.shape)
        the_avg_x = self.avg(the_x).view(B, self.dim)
        the_max_x = self.max(the_x).view(B, self.dim)
        the = torch.concat((the_avg_x, the_max_x), dim=1)
        # B, 4C
        x = torch.concat((rgb, the), dim=1)
        x = self.mlp(x).view(B, self.dim*2, 1)
        # 2, B, C, 1, 1
        channel_rect = x.reshape(B, 2, self.dim, 1, 1).permute(1, 0, 2, 3, 4)
        return channel_rect

class SpatialRect(nn.Module):
    def __init__(self, dim, reduction=1) -> None:
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Conv2d(self.dim*2, self.dim // reduction, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(self.dim // reduction, 2, kernel_size=1),
            nn.Sigmoid()
            # nn.ReLU()
        )
    
    def forward(self, rgb_x, the_x):
        B, _, H, W = rgb_x.shape
        x = torch.cat((rgb_x, the_x), dim=1)
        # 2, B, 1, H, W
        spatial_rect = self.mlp(x).view(B, 2, 1, H, W).permute(1, 0, 2, 3, 4)
        return spatial_rect

class FeatureCorrectModule(nn.Module):
    def __init__(self, dim, reduction=1, lambda_c=.5, lambda_s=.5) -> None:
        super().__init__()
        self.lambda_c = lambda_c
        self.lambda_s = lambda_s
        self.channel_rect = ChannelRect(dim=dim, reduction=reduction)
        self.spatial_rect = SpatialRect(dim=dim, reduction=reduction)

    def forward(self, rgb_x, the_x):
        channel_rect = self.channel_rect(rgb_x, the_x)
        spatial_rect = self.spatial_rect(rgb_x, the_x)
        out_x1 = rgb_x + channel_rect[1] * the_x * self.lambda_c + spatial_rect[1] * the_x * self.lambda_s
        out_x2 = the_x + channel_rect[0] * rgb_x * self.lambda_c + spatial_rect[0] * rgb_x * self.lambda_s
        return out_x1, out_x2

# class WindowAttention(nn.Module):
#     def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0.5, proj_drop=0.5):
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
#         # torch.meshgrid():创建一个坐标网格,返回两个二维张量，分别表示网格中每个点的行坐标和咧坐标
#         coords = torch.stack(torch.meshgrid([coords_h, coords_w])) # 2, Wh, Ww
#         coords_flatten = torch.flatten(coords, 1) # 2, Wh, Ww
#         # 计算相对坐标，虽然维度完全一样但不能直接进行相减运算，需要进行广播操作，否则会得到所有元素都为0的张量
#         relative_coords = coords_flatten[:, :, None] - coords_flatten[: ,None ,:] # 2, Wh*Ww, Wh, Ww
#         relative_coords = relative_coords.permute(1, 2, 0).contiguous() # Wh*Ww, Wh*Ww, 2
#         # 确保每个位置上的相对位置索引是唯一的
#         relative_coords[:, :, 0] += self.window_size[0] - 1
#         relative_coords[:, :, 1] += self.window_size[1] - 1
#         relative_coords[:, :, 0] *= 2 * self.window_size[0] - 1
#         relative_position_index = relative_coords.sum(-1) # Wh*Ww
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
