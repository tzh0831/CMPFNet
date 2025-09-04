from torch import nn
from timm.models.layers import to_2tuple, DropPath, trunc_normal_
import math
import torch
import sys
sys.path.append('/workspace/experiment2')
from model.strength import FeatureCorrectModule as FCM
from model.strength import Weighted
from model.attention import CrossWindow as CW
from util.logger import get_logger
import torch.utils.model_zoo as model_zoo
from functools import partial
import torch.nn.functional as F

model_urls = {
    # 'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    # 'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    # 'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    # 'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

logger = get_logger()
class Backbone(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        layers = [3, 4, 6, 3]
        block  = Bottleneck
        # rgb image branch
        self.inplanes = 64
        # resnet18 conv1
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        # self.relu = nn.ReLU(inplace=False)
        self.relu = nn.LeakyReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # resnet50 conv2_x
        self.layer1 = self._make_layer(block, 64, layers[0])
        # self.conv_layer2 = nn.Conv2d(128, 256, kernel_size=1)
        # resnet50 conv3_x
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        # self.conv_layer3 = nn.Conv2d(320, 512, kernel_size=1)
        # resnet50 conv4_x
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        # self.conv_layer4 = nn.Conv2d(512, 1024, kernel_size=1)
        # resnet50 conv5_x
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # thermal image branch(number of channels=1)
        self.inplanes = 64

        # SegFormer
        self.layer_t = _make_layer_trans()
        # # 第三层rgb向thermal看齐
        # self.rgb_conv = nn.Conv2d(256, 320, kernel_size=1)
        # # 融合后输入第四层时再降回来
        # self.rgb_down = nn.Conv2d(320, 256, kernel_size=1)
        #第三层thermal向rgb看齐
        self.thermal_conv = nn.Sequential(
            nn.Conv2d(320, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU()
        )
        #融合后输入第三层时再升回去
        self.thermal_up = nn.Sequential(
            nn.Conv2d(256, 320, kernel_size=1),
            nn.BatchNorm2d(320),
            nn.LeakyReLU()
        )
        # 融合前thermal通道向rgb对齐
        # self.the_conv1 = nn.Conv2d(64, 256, kernel_size=1)
        # self.the_conv2 = nn.Conv2d(128, 512, kernel_size=1)
        # self.the_conv3 = nn.Conv2d(320, 1024, kernel_size=1)
        # self.the_conv4 = nn.Conv2d(512, 2048, kernel_size=1)
        # thermal图像降采样
        # self.maxpool_x2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.conv_the1 = nn.Conv2d(256, 64, kernel_size=1)
        # self.maxpool_x3 = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.conv_the2 = nn.Conv2d(512, 128, kernel_size=1)
        # self.maxpool_x4 = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.conv_the3 = nn.Conv2d(1024, 320, kernel_size=1)
        # self.maxpool_x5 = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.conv_the4 = nn.Conv2d(2048, 512, kernel_size=1)

        # self.maxpool_x3 = nn.MaxPool2d()
        # merge
        self.merge = RGBTheMerge()

        # edegGenerator
        # 需要传入的是红外图像最低两个层次的特征输出
        self.edge = EdgeGenerator()

        # FeatureAugmentMoudle
        # self.fam = FeatureAugment(dim=512)
        self.fam = FeatureAugment()
        # 为了整除窗口大小，将H进行下采样
        # self.fam_down = nn.AdaptiveAvgPool2d((5, 10))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        if pretrained:
            self._load_resnet_pretrained()

    def forward(self, rgb_x, the_x):
        out=[]
        # 检查一下是否返回四个值(self.layer_t等于的那个函数最终返回了四个值，但是用4个self.layer*_t报错，看一下是不是得后面具体调用的时候才把这四个值分开)
        # print("检查{}".format(len(self.layer_t(the_x))))

        # 检查每一层的维度，看一下融合模块的维度要怎么写
        # print(rgb_x.shape)
        rgb_x = self.conv1(rgb_x)
        # print(the_x.shape)
        rgb_x = self.bn1(rgb_x)
        rgb_x = self.relu(rgb_x)
        rgb_x = self.maxpool(rgb_x)
        the_x = self.layer_t(the_x, layer=1)
        # print(the_x1.shape)
        # print(rgb_x.shape)
        # rgb_x = self.layer1(rgb_x)
        # layer1
        rgb_x = self.layer1(rgb_x)
        # print(rgb_x.shape)
        # 通道对齐
        # the_x1 = self.the_conv1(the_x1)
        # merge
        merge1 = self.merge(rgb_x, the_x, layer=1)
        # rgb_merge1, the_merge1, merge1 = self.merge(rgb_x, the_x, layer=1)
        out.append(merge1)
        
        # print(the_x1.shape)
        # layer2
        the_x2 = self.layer_t(the_x, layer=2)
        rgb_x2 = self.layer2(rgb_x)
        # the_x2 = self.layer_t(the_merge1, layer=2)
        # rgb_x2 = self.layer2(rgb_merge1)

        # merge
        merge2 = self.merge(rgb_x2, the_x2, layer=2)
        # rgb_merge2, the_merge2, merge2 = self.merge(rgb_x2, the_x2, layer=2)
        out.append(merge2)
        # edgeGenerator bc, 64, 120, 160

        # edge = self.edge(rgb_x, rgb_x2)

        edge = self.edge(the_x, the_x2)

        # layer3
        # the_merge2 = self.conv_the2(the_merge2)
        # edge = self.edge(rgb_x_, rgb_x2)
        # edge = self.edge(rgb_x, rgb_x2)
        # rgb_x = self.layer2(rgb_merge1)
        the_x3 = self.layer_t(the_x2, layer=3)
        # the_x3 = self.layer_t(the_merge2, layer=3)
        the_x3 = self.thermal_conv(the_x3)
        rgb_x3 = self.layer3(rgb_x2)
        # rgb_x3 = self.layer3(rgb_merge2)
        # the_x3 = self.the_conv3(the_x3)
        # merge
        merge3 = self.merge(rgb_x3, the_x3, layer=3)
        # rgb_merge3, the_merge3, merge3 = self.merge(rgb_x3, the_x3, layer=3)
        out.append(merge3)

        # rgb_fam = self.fam(rgb_x3)
        # the_fam = self.fam(the_x3)
        # # rgb_fam = self.fam(rgb_merge3)
        # # the_fam = self.fam(the_merge3)

        # merge4 = self.merge(rgb_fam, the_fam, layer=4)
        # out.append(merge4)

        
        the_x3 = self.thermal_up(the_x3)
        the_x4 = self.layer_t(the_x3, layer=4)
        # the_x3 = self.thermal_up(the_fam)
        # the_x4 = self.layer_t(the_x3, layer=4)
        # the_x4 = self.layer_t(the_merge4, layer=4)
        rgb_x4 = self.layer4(rgb_x3)
        # rgb_x4 = self.layer4(rgb_merge4)
        merge4 = self.merge(rgb_x4, the_x4, layer=4)
        # rgb_merge4, the_merge4, merge4 = self.merge(rgb_x4, the_x4, layer=4)
        out.append(merge4)
        # layer4
        # the_merge3 = self.conv_the3(the_merge3)
        # rgb_x = self.layer3(rgb_merge2)
        # rgb_merge3 = self.conv_layer3(rgb_merge3)
        # rgb_x = self.layer3(rgb_merge3)
        # rgb_merge3 = self.rgb_down(rgb_merge3)
        
        rgb_fam = self.fam(rgb_x4)
        the_fam = self.fam(the_x4)
        # rgb_fam = self.fam(rgb_merge3)
        # the_fam = self.fam(the_merge3)

        merge5 = self.merge(rgb_fam, the_fam, layer=5)
        out.append(merge5)

        # the_x4 = self.layer_t(the_merge3, layer=4)
        # rgb_merge3 = self.rgb_down(rgb_merge3)
        # rgb_x4 = self.layer4(rgb_merge3)
        # # the_x4 = self.the_conv4(the_x4)
        # rgb_merge4, the_merge4, merge4 = self.merge(rgb_x4, the_x4, layer=4)
        # out.append(merge4)

        # FAM bc, 2048, 7, 9
        # rgb_fam = self.fam(rgb_merge4)
        # the_fam = self.fam(the_merge4)

        # rgb_fam = self.fam_down(rgb_fam)
        # the_fam = self.fam_down(the_fam)
        # merge
        # merge5 = self.merge(rgb_fam, the_fam, layer=5)
        # out.append(merge5)

        # rgb_merge4 = self.conv_layer4(rgb_merge4)
        # rgb_x = self.layer4(rgb_merge4)
        # rgb_x = self.conv_x5(rgb_x)
        
        # merge
        # merge5 = self.merge(rgb_fam, the_fam, layer=5)
        # out.append(merge5)
        
        return out, edge

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)
    
    def _load_resnet_pretrained(self):
        pretrain_dict = model_zoo.load_url(model_urls['resnet34'])
        pretrain_t_dict = torch.load('/workspace/experiment2/pretrained/segformer_b4_weights.pth')
        model_dict = {}
        new_dict = {}
        new_t_dict = {}
        # 获取当前模型的状态字典
        state_dict = self.state_dict()
        for k, v in pretrain_dict.items():
            if k.find('running_mean') == -1 and k.find('running_var') == -1:
                new_dict[k] = v
        for k, v in pretrain_t_dict.items():
            if k.find('block2.') != -1 and int(k[7]) > 3:
                continue
            if k.find('block3.') != -1:
                if k[8] != '.':
                    continue
                else:
                    if int(k[7]) > 5:
                        continue
            if k.find('attn.q.bias') == -1 and k.find('attn.kv.bias') == -1 and k.find('head'):
                new_t_dict[k] = v

        for k, v in new_dict.items():
            if k in state_dict:
                if k.startswith('conv1'):
                    model_dict[k] = v
                    # model_dict[k.replace('conv1', 'conv1_d')] = v
                elif k.startswith('bn1'):
                    model_dict[k] = v
                    # model_dict[k.replace('bn1', 'bn1_d')] = v
                elif k.startswith('layer'):
                    model_dict[k] = v
                    # layer后面还跟着数字区分层，所以:6
                    # model_dict[k[:6]+'_d'+k[6:]] = v
                    # model_dict[k[:6]+'_m'+k[6:]] = v
        for k, v in new_t_dict.items():
            model_dict['layer_t.extra_' + k] = v
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)
        
class _make_layer_trans(nn.Module):
    def __init__(self, img_size=224, in_chans=3, num_classes=5, embed_dims=[64, 128, 320, 512], 
                 num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=True, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0.1, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 8, 27, 3], sr_ratios=[8, 4, 2, 1]):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        # patch_embed
        self.extra_patch_embed1 = OverlapPatchEmbed(img_size=img_size, patch_size=7, stride=4, in_chans=in_chans,
                                              embed_dim=embed_dims[0])
        self.extra_patch_embed2 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0],
                                              embed_dim=embed_dims[1])
        self.extra_patch_embed3 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1],
                                              embed_dim=embed_dims[2])
        self.extra_patch_embed4 = OverlapPatchEmbed(img_size=img_size // 16, patch_size=3, stride=2, in_chans=embed_dims[2],
                                              embed_dim=embed_dims[3])

        # transformer encoder
        # 随机梯度衰减
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        self.extra_block1 = nn.ModuleList([Block(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])
            for i in range(depths[0])])
        self.extra_norm1 = norm_layer(embed_dims[0])
        cur += depths[0]

        self.extra_block2 = nn.ModuleList([Block(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1])
            for i in range(depths[1])])
        self.extra_norm2 = norm_layer(embed_dims[1])
        cur += depths[1]

        self.extra_block3 = nn.ModuleList([Block(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[2])
            for i in range(depths[2])])
        self.extra_norm3 = norm_layer(embed_dims[2])
        cur += depths[2]

        self.extra_block4 = nn.ModuleList([Block(
            dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[3])
            for i in range(depths[3])])
        self.extra_norm4 = norm_layer(embed_dims[3])
        cur += depths[3]
    
    def forward(self, the_x, layer):
        B = the_x.shape[0]
        the_end = the_x
        # layer1
        if layer == 1:
            the_x1, H, W = self.extra_patch_embed1(the_x)
            for i, blk in enumerate(self.extra_block1):
                the_x1 = blk(the_x1, H, W)
            the_x1 = self.extra_norm1(the_x1)
            the_x1 = the_x1.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            the_end = the_x1
        # layer2
        if layer == 2:
            the_x2, H, W = self.extra_patch_embed2(the_x)
            for i, blk in enumerate(self.extra_block2):
                the_x2 = blk(the_x2, H, W)
            the_x2 = self.extra_norm2(the_x2)
            the_x2 = the_x2.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            the_end = the_x2
        # layer3
        if layer == 3:
            the_x3, H, W = self.extra_patch_embed3(the_x)
            for i, blk in enumerate(self.extra_block3):
                the_x3 = blk(the_x3, H, W)
            the_x3 = self.extra_norm3(the_x3)
            the_x3 = the_x3.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            the_end = the_x3
        # layer4
        if layer == 4:
            the_x4, H, W = self.extra_patch_embed4(the_x)
            for i, blk in enumerate(self.extra_block4):
                the_x4 = blk(the_x4, H, W)
            the_x4 = self.extra_norm4(the_x4)
            the_x4 = the_x4.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            the_end = the_x4
        return the_end

class RGBTheMerge(nn.Module):
    def __init__(self, num_classes=5, embed_dims=[64, 128, 256, 512], num_heads=[1, 2, 4, 8]):
        super().__init__()
        self.num_classes = num_classes
        self.weighted = nn.Sequential(
            Weighted(dim=embed_dims[0]),
            Weighted(dim=embed_dims[1]),
            Weighted(dim=embed_dims[2]),
            Weighted(dim=embed_dims[3]),
            Weighted(dim=embed_dims[3]),
        )
        self.FCMs = nn.ModuleList((
            FCM(dim=embed_dims[0], reduction=1),
            FCM(dim=embed_dims[1], reduction=1),
            FCM(dim=embed_dims[2], reduction=1),
            FCM(dim=embed_dims[3], reduction=1),
            FCM(dim=embed_dims[3], reduction=1)
        ))
        self.attention = nn.ModuleList((
            CW(dim=embed_dims[0], num_heads=num_heads[0]),
            CW(dim=embed_dims[1], num_heads=num_heads[1]),
            CW(dim=embed_dims[2], num_heads=num_heads[2]),
            CW(dim=embed_dims[3], num_heads=num_heads[3]),
            CW(dim=embed_dims[3], num_heads=num_heads[3]),
            # CW(dim=embed_dims[0], depth=2, num_heads=num_heads[0], window_size=8, input=(120, 160)),
            # CW(dim=embed_dims[1], depth=2, num_heads=num_heads[1], window_size=5, input=(60, 80)),
            # CW(dim=embed_dims[2], depth=2, num_heads=num_heads[2], window_size=5, input=(30, 40)),
            # CW(dim=embed_dims[3], depth=2, num_heads=num_heads[3], window_size=5, input=(15, 20)),
            # CW(dim=embed_dims[3], depth=2, num_heads=num_heads[3], window_size=5, input=(15, 20))
        ))
        self.apply(self._init_weights)
    
    # 初始化阶段被调用，用于初始化模型中的各个层
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
            # fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            # fan_out //= m.groups
            # m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
    
    def forward(self, rgb_x, the_x, layer):
        # print(self.attention)
        if layer == 1:
            # 2, 256, 120, 160
            rgb_rect1, the_rect1 = self.weighted[0](rgb_x, the_x)
            rgb_mid1, the_mid1 = self.FCMs[0](rgb_rect1, the_rect1)
            out1 = self.attention[0](rgb_mid1, the_mid1)
            # rgb_mid1, the_mid1 = self.FCMs[0](rgb_x, the_x)
            # rgb_rect1, the_rect1 = self.weighted[0](rgb_mid1, the_mid1)
            # out1 = self.attention[0](rgb_rect1, the_rect1)
            return out1
            # return rgb_mid1, the_mid1, out1
            # return rgb_rect1, the_rect1, out1
        elif layer == 2:
            rgb_rect2, the_rect2 = self.weighted[1](rgb_x, the_x)
            rgb_mid2, the_mid2 = self.FCMs[1](rgb_rect2, the_rect2)
            out2 = self.attention[1](rgb_mid2, the_mid2)
            # rgb_mid2, the_mid2 = self.FCMs[1](rgb_x, the_x)
            # rgb_rect2, the_rect2 = self.weighted[1](rgb_mid2, the_mid2)
            # out2 = self.attention[1](rgb_rect2, the_rect2)
            return out2
            # return rgb_mid2, the_mid2, out2
            # return rgb_rect2, the_rect2, out2
        elif layer == 3:
            rgb_rect3, the_rect3 = self.weighted[2](rgb_x, the_x)
            rgb_mid3, the_mid3 = self.FCMs[2](rgb_rect3, the_rect3)
            out3 = self.attention[2](rgb_mid3, the_mid3)
            # rgb_mid3, the_mid3 = self.FCMs[2](rgb_x, the_x)
            # rgb_rect3, the_rect3 = self.weighted[2](rgb_mid3, the_mid3)
            # out3 = self.attention[2](rgb_rect3, the_rect3)
            return out3
            # return rgb_mid3, the_mid3, out3
            # return rgb_rect3, the_rect3, out3
        elif layer == 4:
            rgb_rect4, the_rect4 = self.weighted[3](rgb_x, the_x)
            rgb_mid4, the_mid4 = self.FCMs[3](rgb_rect4, the_rect4)
            out4 = self.attention[3](rgb_mid4, the_mid4)
            # rgb_mid4, the_mid4 = self.FCMs[3](rgb_x, the_x)
            # rgb_rect4, the_rect4 = self.weighted[3](rgb_mid4, the_mid4)
            # out4 = self.attention[3](rgb_rect4, the_rect4)
            return out4
            # return rgb_mid4, the_mid4, out4
            # return out4
            # return rgb_rect4, the_rect4, out4
        elif layer == 5:
            rgb_rect5, the_rect5 = self.weighted[4](rgb_x, the_x)
            rgb_mid5, the_mid5 = self.FCMs[4](rgb_rect5, the_rect5)
            out5 = self.attention[4](rgb_mid5, the_mid5)
            # rgb_mid5, the_mid5 = self.FCMs[4](rgb_x, the_x)
            # rgb_rect5, the_rect5 = self.weighted[4](rgb_mid5, the_mid5)
            # out5 = self.attention[4](rgb_rect5, the_rect5)
            return out5

class Bottleneck(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        # self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        # self.bn3 = nn.BatchNorm2d(planes * 4)
        # self.relu = nn.ReLU(inplace=False)
        self.relu = nn.LeakyReLU()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        shortcut = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        # out = self.conv3(out)
        # out = self.bn3(out)
        
        if self.downsample is not None:
            shortcut = self.downsample(x)
        out = out + shortcut
        out = self.relu(out)
        return out

class DWConv(nn.Module):
    """
    Depthwise convolution bloc: input: x with size(B N C); output size (B N C)
    """
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.permute(0, 2, 1).reshape(B, C, H, W).contiguous() # B N C -> B C N -> B C H W
        x = self.dwconv(x) 
        x = x.flatten(2).transpose(1, 2) # B C H W -> B N C
        return x

class Block(nn.Module):
    """
    Transformer Block: Self-Attention -> Mix FFN -> OverLap Patch Merging
    """
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
            # fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            # fan_out //= m.groups
            # m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        # self-attention
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        # mix-ffn
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        """
        MLP Block: 
        """
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
            # fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            # fan_out //= m.groups
            # m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        # x = self.drop(x)
        x = self.fc2(x)
        # x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=4, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # Linear embedding
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
            # fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            # fan_out //= m.groups
            # m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x.shape
        # B N C -> B N num_head C//num_head -> B C//num_head N num_heads
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3) 
        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W) 
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1) 
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4) 
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4) 
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
class OverlapPatchEmbed(nn.Module):
    """ 
    Image to Patch Embedding
    """
    def __init__(self, img_size, patch_size, stride, in_chans, embed_dim):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        # if isinstance(m, nn.Linear):
        #     trunc_normal_(m.weight, std=.02)
        #     if isinstance(m, nn.Linear) and m.bias is not None:
        #         #偏置初始化为0
        #         nn.init.constant_(m.bias, 0)
        if isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
            # fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            # fan_out //= m.groups
            # m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        # B C H W
        
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        # B H*W/16 C
        x = self.norm(x)
        return x, H, W

class EdgeGenerator(nn.Module):
    def __init__(self, dim=[64, 128], out=32) -> None:
        super().__init__()
        # self.BLRConv_96_ = nn.Sequential(
        #     nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(dim),
        #     nn.LeakyReLU()
        # )
        # self.BLRConv_96 = nn.Sequential(
        #     nn.Conv2d(dim // 2, dim, kernel_size=1),
        #     nn.BatchNorm2d(dim),
        #     nn.LeakyReLU()
        # )
        self.BLRConv_64 = nn.Conv2d(dim[0], out, kernel_size=1)
        self.BLRConv_64_ = nn.Sequential(
            nn.Conv2d(out, out, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out),
            nn.LeakyReLU()
        )
        self.BLRConv_128 = nn.Conv2d(dim[1], out, kernel_size=1)
        self.BLRConv_128_ = nn.Sequential(
            nn.Conv2d(out, out, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out),
            nn.LeakyReLU()
        )
        self.pool = nn.Conv2d(dim[1], dim[0], kernel_size=1)
        self.pool_ = nn.Sequential(
            nn.Conv2d(dim[0], dim[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dim[0]),
            nn.LeakyReLU()
        )
        # self.edge_conv = nn.Conv2d(dim[0], dim[0], kernel_size=3, stride=1, padding=1)
        # self.edge_up = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        # self.BLRConv_192 = nn.Conv2d(dim, dim // 2, kernel_size=1)
        # self.BLRConv_192_ = nn.Sequential(
        #     nn.Conv2d(dim // 2, dim // 2, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(dim // 2),
        #     nn.LeakyReLU()
        # )
        # self.BLRConv_288  = nn.Conv2d(288, 128, kernel_size=1)
        # self.BLRConv_288_ = nn.Sequential(
        #     nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(128),
        #     nn.LeakyReLU()
        # )
        # self.BLRConv_128  = nn.Conv2d(256, 128, kernel_size=1)
        # self.BLRConv_128_ = nn.Sequential(
        #     nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(128),
        #     nn.LeakyReLU()
        # )
        # self.conv1 = nn.Conv2d(dim * 2, dim, kernel_size=1)
        # self.conv2 = nn.Conv2d(dim // 2, 64, kernel_size=1)
        # self.conv3 = nn.Conv2d(dim // 4, 64, kernel_size=1)
        # self.conv3 = nn.Conv2d(256, 128, kernel_size=1)
        self.conv = nn.Conv2d(128, 64, kernel_size=1)
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        # self.conv = nn.Conv2d(dim, dim, kernel_size=1)
    #     self.apply(self._init_weights)
    
    # def _init_weights(self, m):
    #     if isinstance(m, nn.Conv2d):
    #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
    #         m.weight.data.normal_(0, math.sqrt(2. / n))
    #     elif isinstance(m, nn.BatchNorm2d):
    #         m.weight.data.fill_(1)
    #         m.bias.data.zero_()
    def forward(self, x1, x2):
        """
        x1和x2来自thermal图
        x1:第一层（最低层）的特征输出(thermal)
        x2:第二层的特征输出(thermal)
        """
        # 先把x2进行上采样操作，使其宽高与第一层一致后再进行通道拼接。
        # （选择x2上采样而不是x1下采样的原因：越底层的特征数据越关注局部信息，更有利于边缘的识别）
        _, _, H, W = x1.shape
        edge1 = self.BLRConv_64(x1)
        edge1 = self.BLRConv_64_(edge1)
        edge2 = self.BLRConv_128(x2)
        edge2 = self.BLRConv_128_(edge2)
        edge2 = F.interpolate(edge2, size=(H, W), mode='bilinear', align_corners=True)
        # edge2 = self.edge_up(edge2)
        edge = torch.concat((edge1, edge2), dim=1)
        # edge = self.edge_conv(edge)
        # x2 = F.interpolate(x2, size=(H, W), mode='bilinear')
        # x = torch.concat((x1, x2), dim=1)
        # mid_x1 = self.BLRConv_192(x)
        # mid_x1 = self.BLRConv_192_(mid_x1)
        # mid_x2 = self.BLRConv_96(mid_x1)
        # mid_x2 = self.BLRConv_96_(mid_x2)
        
        # mid_x = torch.concat((mid_x1, mid_x2), dim=1)
        # mid_x = self.BLRConv_288(mid_x)
        # mid_x = self.BLRConv_288_(mid_x)
        # mid_x = self.conv1(mid_x)
        # mid_x = self.BLRConv_384(mid_x)
        # mid_x = self.conv2(mid_x)
        # mid_x = x1 + mid_x
        max_x = self.maxpool(edge)
        avg_x = self.avgpool(edge)
        pool_x = torch.concat((max_x, avg_x), dim=1)
        pool_x = self.pool(pool_x)
        pool_x = self.pool_(pool_x)
        mul_x = torch.mul(pool_x, edge)
        x = edge + mul_x
        # x = self.conv(x)
        return x

class FeatureAugment(nn.Module):
    def __init__(self, dim=512, dilation=[6, 12, 18]) -> None:
        super().__init__()
        # self.conv = nn.Conv2d(dim, dim // 2, kernel_size=3, stride=2, padding=1)
        self.conv = nn.Conv2d(dim, dim // 2, kernel_size=1)
        self.conv3_r2 = nn.Sequential(
            nn.Conv2d(dim, dim // 2, kernel_size=3, padding=6, dilation=dilation[0]),
            nn.BatchNorm2d(dim // 2),
            nn.LeakyReLU()
        )
        self.gamma2 = nn.Parameter(torch.zeros(1))
        self.conv3_r4 = nn.Sequential(
            nn.Conv2d(dim, dim // 2, kernel_size=3, padding=12, dilation=dilation[1]),
            nn.BatchNorm2d(dim // 2),
            nn.LeakyReLU()
        )
        self.gamma4 = nn.Parameter(torch.zeros(1))
        self.conv3_r6 = nn.Sequential(
            nn.Conv2d(dim, dim // 2, kernel_size=3, padding=18, dilation=dilation[2]),
            nn.BatchNorm2d(dim // 2),
            nn.LeakyReLU()
        )
        self.gamma6 = nn.Parameter(torch.zeros(1))
        self.query = nn.Conv2d(dim // 2, dim // 8, kernel_size=1)
        self.key = nn.Conv2d(dim // 2, dim // 8, kernel_size=1)
        self.value = nn.Conv2d(dim // 2, dim // 2, kernel_size=1)
        self.activation = nn.Softmax(dim=-1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.pool_conv = nn.Sequential(
            nn.Conv2d(dim, dim // 2, kernel_size=1),
            nn.LeakyReLU()
        )
        # self.attn = channel_attention(dim // 2)
        self.down = nn.Sequential(
            nn.Conv2d((dim // 2) * 5, dim, kernel_size=1),
            nn.BatchNorm2d(dim),
            nn.LeakyReLU()
        )
        # self.down1 = nn.Sequential(
        #     nn.Conv2d(dim * 5, 1024, kernel_size=1),
        #     nn.BatchNorm2d(1024),
        #     nn.LeakyReLU()
        # )
        # self.down2 = nn.Sequential(
        #     nn.Conv2d(1024, dim, kernel_size=1),
        #     nn.BatchNorm2d(dim),
        #     nn.LeakyReLU()
        # )
        # self.attn_r4 = channel_attention(dim)
        # self.attn_r6 = channel_attention(dim)

    # def channel_attention(self, dim, reduction=4):
    #     avgpool = nn.AdaptiveAvgPool2d(1)
    #     avgpool_view = avgpool.view(4, dim)
    #     # conv = nn.Conv2d(dim, dim, kernel_size=1)
    #     fc1 = nn.Linear(dim, dim // reduction)
    #     bn1 = nn.BatchNorm2d(dim // reduction)
    #     fc2 = nn.Linear(dim // reduction, dim)
    #     bn2 = nn.BatchNorm2d(dim)
    #     activation = nn.Sigmoid()
    #     # activation = nn.ReLU()
    #     return nn.Sequential(*[avgpool, avgpool_view, fc1, bn1, fc2, bn2, activation])
    
    def forward(self, x):
        # bc, C, 7, 9
        x1 = self.conv(x)
        x2 = self.conv3_r2(x)
        # bc, C, 7, 9
        # x2 = self.attn(x2)
        bc, c, h, w = x2.size()
        query = self.query(x2).view(bc, -1, w * h).permute(0, 2, 1)
        key = self.key(x2).view(bc, -1, w * h)
        energy = torch.bmm(query, key)
        attn = self.activation(energy)
        value = self.value(x2).view(bc, -1, w * h)
        x2_ = torch.bmm(value, attn.permute(0, 2, 1)).view(bc, c, h, w)
        x2_ = self.gamma2 * x2_ + x2 
        x3 = self.conv3_r4(x)
        # C, 8, 10
        # x3 = self.attn(x3)
        # bc, c, h, w = x3.size()
        query = self.query(x3).view(bc, -1, w * h).permute(0, 2, 1)
        key = self.key(x3).view(bc, -1, w * h)
        energy = torch.bmm(query, key)
        attn = self.activation(energy)
        value = self.value(x3).view(bc, -1, w * h)
        x3_ = torch.bmm(value, attn.permute(0, 2, 1)).view(bc, c, h, w)
        x3_ = self.gamma4 * x3_ + x3 
        x4 = self.conv3_r6(x)
        # x4 = self.attn(x4)
        # bc, c, h, w = x4.size()
        query = self.query(x4).view(bc, -1, w * h).permute(0, 2, 1)
        key = self.key(x4).view(bc, -1, w * h)
        energy = torch.bmm(query, key)
        attn = self.activation(energy)
        value = self.value(x4).view(bc, -1, w * h)
        x4_ = torch.bmm(value, attn.permute(0, 2, 1)).view(bc, c, h, w)
        x4_ = self.gamma6 * x4_ + x4 
        x5 = self.pool_conv(self.avgpool(x))
        x5 = nn.functional.interpolate(x5, (15, 20), mode='bilinear')
        # x5 = nn.functional.interpolate(x5, (23, 40), mode='bilinear')
        # x = x1 + x2 + x3 + x4 + x5
        x = torch.concat((x1, x2_, x3_, x4_, x5), dim=1)
        x = self.down(x)
        # x = self.down2(self.down1(x))
        # reduced = self.down(x)
        # x = self.attn(reduced)
        # x = x * reduced + reduced
        # x = self.conv_e(self.avgpool_e(x))
        # x = torch.mul(x, edge)
        return x

# class channel_attention(nn.Module):
#     def __init__(self, dim, reduction=4):
#         super().__init__()
#         self.avgpool = nn.AdaptiveAvgPool2d(1)
#         # conv = nn.Conv2d(dim, dim, kernel_size=1)
#         self.fc1 = nn.Linear(dim, dim // reduction)
#         self.relu = nn.ReLU()
#         self.fc2 = nn.Linear(dim // reduction, dim)
#         # self.bn2 = nn.BatchNorm2d(dim)
#         self.activation = nn.Sigmoid()
    
#     def forward(self, x):
#         bc, dim, _, _ = x.size()
#         out = self.avgpool(x)
#         out = out.view(bc, dim)
#         out = self.relu(self.fc1(out))
#         out = self.relu(self.fc2(out))
#         out = self.activation(out).view(bc, dim, 1, 1)
#         out = out * x
#         return out

# class mit_b4(_make_layer_trans):
#     def __init__(self, fuse_cfg=None, **kwargs):
#         super(mit_b4, self).__init__(
#             patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
#             qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 8, 27, 3], sr_ratios=[8, 4, 2, 1],
#             drop_rate=0.0, drop_path_rate=0.1)
    
# if __name__ == '__main__':
#     model = Backbone()
#     for name, value in model.named_parameters():
#         print('name:{}'.format(name))