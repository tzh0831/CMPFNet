from torch import nn
import torch
from model.encoder import Backbone

class Decoder(nn.Module):
    def __init__(self, num_class):
        super().__init__()
        # self.maxpool = nn.AdaptiveMaxPool2d((120, 160))
        # self.avgpool = nn.AdaptiveAvgPool2d((120, 160)
        # self.conv = nn.ModuleList((
        #     # nn.Sequential(nn.Conv2d(512, 320, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(320), nn.LeakyReLU()),
        #     # nn.Sequential(nn.Conv2d(320, 128, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(128), nn.LeakyReLU()),
        #     # nn.Sequential(nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(64), nn.LeakyReLU()),
        #     nn.Sequential(nn.Conv2d(512, 320, kernel_size=1), nn.BatchNorm2d(320), nn.LeakyReLU()),
        #     nn.Sequential(nn.Conv2d(320, 128, kernel_size=1), nn.BatchNorm2d(128), nn.LeakyReLU()),
        #     nn.Sequential(nn.Conv2d(128, 64, kernel_size=1), nn.BatchNorm2d(64), nn.LeakyReLU()),
        #     # nn.Conv2d(512, 320, kernel_size=1),
        #     # nn.Conv2d(320, 128, kernel_size=1),
        #     # nn.Conv2d(128, 64, kernel_size=1)
        # ))
        self.up_conv = nn.ModuleList((
            # nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(512), nn.LeakyReLU()),
            nn.Sequential(nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(256), nn.LeakyReLU()),
            nn.Sequential(nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(128), nn.LeakyReLU()),
            nn.Sequential(nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(64), nn.LeakyReLU())
            # nn.Sequential(nn.Conv2d(320, 320, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(320), nn.LeakyReLU()),
            # nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(128), nn.LeakyReLU()),
            # nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(64), nn.LeakyReLU()),
        ))
        self.layer1 = mid(512)
        self.layer2 = mid(256)
        self.layer3 = mid(128)
        self.layer4 = mid(64)
        # self.down = nn.ModuleList((
        #     nn.Conv2d(512, 320, kernel_size=1),
        #     nn.Conv2d(320, 128, kernel_size=1),
        #     nn.Conv2d(128, 64, kernel_size=1)
        # ))
        # self.conv1 = nn.Conv2d(1472, 512, kernel_size=1)
        # self.conv2 = nn.Conv2d(512, 64, kernel_size=1)
        self.sig = nn.Sigmoid()
        self.outconv = nn.Conv2d(64 * 2, num_class, kernel_size=3, stride=1, padding=1, bias=False)
        # self.out_up = nn.Conv2d(num_class, num_class, kernel_size=3, stride=1, padding=1)
        # self.drop = nn.Dropout2d(0.1)
        self.outconv_edge = nn.Conv2d(64, 2, kernel_size=3, stride=1, padding=1, bias=False)
        # self.edge_up = nn.Conv2d(2, 2, kernel_size=3, stride=1, padding=1)
        self.backbone = Backbone()
        self.dropout = nn.Dropout2d(0.1)
        # self.fea = FeatureAugment(pool=(60, 80), dim=64)
        # self.soft = nn.Softmax(dim=1)

    def forward(self, x, edge):
        x0, x1, x2, x3, x4 = x
        # x = x0 + x1 + x2 + x3 + x4
        # bc, _, _ = x0.shape
        # x0 = x0.reshape(bc, 120, 160, 64).permute(0, 3, 1, 2)
        # x1 = x1.reshape(bc, 60, 80, 128).permute(0, 3, 1, 2)
        # x1 = nn.functional.interpolate(x1, (15, 20), mode='bilinear')
        # x2 = x2.reshape(bc, 30, 40, 320).permute(0, 3, 1, 2)
        # x2 = nn.functional.interpolate(x2, (15, 20), mode='bilinear')
        # x3 = x3.reshape(bc, 15, 20, 512).permute(0, 3, 1, 2)
        # x4 = x4.reshape(bc, 5, 10, 512).permute(0, 3, 1, 2)
        # x4 = nn.functional.interpolate(x4, (15, 20), mode='bilinear')
        # x = torch.concat((x1, x2, x3, x4), dim=1)
        # x = self.conv1(x)
        # bc, 512, 8, 10
        # x = self.fea(x)
        # x = self.conv2(x)
        # x = nn.functional.interpolate(x, (120, 160), mode='bilinear')
        # x_ = torch.mul(x0, x)
        # max_x = self.maxpool(x_)
        # avg_x = self.avgpool(x_)
        # x_mid = torch.concat((max_x, avg_x), dim=1)
        # left_x = self.BLRConv(x_mid)
        # left_x = self.sig(left_x)
        # left_x = torch.mul(left_x, x)
        # left_x = torch.mul(left_x, edge)
        # right_x = 1 - self.sig(self.BLRConv(x_mid))
        # right_x = torch.mul(right_x, x)
        # right_x = torch.mul(right_x, edge)
        # x_edge = self.outconv(left_x + right_x)

        # x = self.conv2(self.conv1(x))
        # x = nn.functional.interpolate(x, (120, 160), mode='bilinear')
        # x_ = torch.mul(x0, x)
        # max_x = self.maxpool(x_)
        # avg_x = self.avgpool(x_)
        # x_mid = torch.concat((max_x, avg_x), dim=1)
        # left_x = self.sig(self.BLRConv(x_mid))
        # left_x = torch.mul(left_x, x)
        # left_x_edge = torch.mul(left_x, edge)
        # right_x = 1 - self.sig(self.BLRConv(x_mid))
        # right_x = torch.mul(right_x, x)
        # right_x_edge = torch.mul(right_x, edge)
        # x_edge = self.outconv(left_x_edge + right_x_edge)
        # x = self.outconv(left_x + right_x)
        

        # # 5个融合结果
        # # x, edge = self.backbone(rgb_x, the_x)
        # x0, x1, x2, x3, x4 = x
        # # 先将x0~x4的信息一层层进行融合，最后再与edge进行融合
        # layer1(x4 x3)
        # PST900(40, 23)
        B, _, C = x4.shape
        # x4 = x4.reshape(B, 23, 40, C).permute(0, 3, 1, 2)
        x4 = x4.reshape(B, 15, 20, C).permute(0, 3, 1, 2)
        # x4 = self.BLRConv[0](x4)

        # x4 = nn.functional.interpolate(x4, (30, 40), mode='bilinear')
        # x4 = self.up_conv[0](x4)

        # x4 = self.up_conv[0](x4)
        # x3 = x3.reshape(B, 23, 40, C).permute(0, 3, 1, 2)
        x3 = x3.reshape(B, 15, 20, 512).permute(0, 3, 1, 2)
        # x3 = x3.reshape(B, 15, 20, C).permute(0, 3, 1, 2)
        
        # x = x4 + x3
        # x_ = x4 * x3
        # x = torch.concat((max_x, avg_x), dim=1)
        # 降维
        # x = self.conv[0](x)
        # left_x = torch.mul(self.sig(x), x4)
        # right_x = torch.mul(1 - self.sig(x), x3)
        # x = left_x + right_x
        x = self.layer1(x4, x3)

        # layer2(layer1's x x2)
        # x_layer1 = self.conv[0](x)
        # x_layer1 = nn.functional.interpolate(x, (45, 80), mode='bilinear')
        x_layer1 = nn.functional.interpolate(x, (30, 40), mode='bilinear')
        x_layer1 = self.up_conv[0](x_layer1)
        # x2 = x2.reshape(B, 45, 80, 256).permute(0, 3, 1, 2)
        x2 = x2.reshape(B, 30, 40, 256).permute(0, 3, 1, 2)
        # x = x_layer1 + x2
        # left_x = torch.mul(self.sig(x), x_layer1)
        # right_x = torch.mul(1 - self.sig(x), x2)
        # left_x = torch.mul(self.sig(x), x_layer1)
        # right_x = torch.mul(1 - self.sig(x), x2)
        # x = left_x + right_x

        # x = self.layer2(x, x2)
        x = self.layer2(x_layer1, x2)

        # layer3(layer2's x x1)
        # x_layer2 = self.conv[1](x)
        # x_layer2 = nn.functional.interpolate(x, (90, 160), mode='bilinear')
        x_layer2 = nn.functional.interpolate(x, (60, 80), mode='bilinear')
        x_layer2 = self.up_conv[1](x_layer2)
        # x1 = x1.reshape(B, 90, 160, 128).permute(0, 3, 1, 2)
        x1 = x1.reshape(B, 60, 80, 128).permute(0, 3, 1, 2)
        # x = x_layer2 + x1
        # left_x = torch.mul(self.sig(x), x_layer2)
        # right_x = torch.mul(1 - self.sig(x), x1)
        # left_x = torch.mul(self.sig(x), x_layer2)
        # right_x = torch.mul(1 - self.sig(x), x1)
        # x = left_x + right_x
        x = self.layer3(x_layer2, x1)

        # layer4
        # x_layer3 = self.conv[2](x)
        # x_layer3 = nn.functional.interpolate(x, (180, 320), mode='bilinear')
        x_layer3 = nn.functional.interpolate(x, (120, 160), mode='bilinear')
        x_layer3 = self.up_conv[2](x_layer3)
        # x0 = x0.reshape(B, 180, 320, 64).permute(0, 3, 1, 2)
        x0 = x0.reshape(B, 120, 160, 64).permute(0, 3, 1, 2)
        # x = x_layer3 + x0
        # 最后要与edge进行融合
        # x = x + edge
        # x = self.fea(x)
        # x = nn.functional.interpolate(x, (120, 160), mode='bilinear')
        # left_x = torch.mul(self.sig(x), x_layer3)
        # # left_x = left_x + edge
        # right_x = torch.mul(1- self.sig(x), x0)
        # # right_x = right_x + edge
        # x_edge = left_x + right_x + edge
        x = self.layer4(x_layer3, x0)
        # x_edge = x + edge

        x_edge = torch.concat((x, edge), dim=1)
        # left_x = torch.mul(torch.mul(self.sig(x), x_layer3), edge)
        # right_x = torch.mul(torch.mul(1 - self.sig(x), x0), edge)
        # x_edge = torch.concat((x_edge, edge), dim=1)
        # x_edge = self.conv[2](x_edge)
        # x_edge = self.fea(x_edge)
        # x_edge = nn.functional.interpolate(x_edge, (720, 1280), mode='bilinear')
        x_edge = nn.functional.interpolate(x_edge, (480, 640), mode='bilinear')
        x_edge = self.dropout(x_edge)
        x_edge = self.outconv(x_edge) # 2, num_class, 120, 160
        # x_edge = self.drop(x_edge)
        # x = nn.functional.interpolate(x, (480, 640), mode='bilinear')
        edge = nn.functional.interpolate(edge, (480, 640), mode='bilinear')
        edge = self.dropout(edge)
        edge = self.outconv_edge(edge)
        # edge = self.drop(edge)
        edge = self.sig(edge)
        
        # x_edge = self.out_up(x_edge)
        
        # edge = self.edge_up(edge)
            # x_edge = nn.functional.interpolate(x_edge, (480, 640))
            # x = self.soft(x)
            # print(x)
        # if not self.training:
        #     x = self.soft(x)
        # return x_edge
        return x_edge, edge

# 不同层特征公用的一个卷积+BN+LeakyReLU模块
# class BLRConv(nn.Module):
#     def __init__(self, in_channel, out_channel):
#         super().__init__()
#         self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1)
#         self.bn = nn.BatchNorm2d(out_channel)
#         self.act = nn.LeakyReLU()
    
#     def forward(self, x):
#         x = self.act(self.bn(self.conv(x)))
#         return x
    
class mid(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.local_global = local_global(channels)
    def forward(self, high_x, low_x):
        x = self.local_global(high_x, low_x)
        return x

class local_global(nn.Module):
    def __init__(self, channels):
        super().__init__()
        mid_channels = channels // 4
        self.local = nn.Sequential(
            nn.Conv2d(channels, mid_channels, kernel_size=1),
            nn.BatchNorm2d(mid_channels),
            nn.LeakyReLU(),
            nn.Conv2d(mid_channels, channels, kernel_size=1),
            nn.BatchNorm2d(channels)
        )
        self.global_ = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, mid_channels, kernel_size=1),
            nn.LeakyReLU(),
            nn.Conv2d(mid_channels, channels, kernel_size=1)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, high_x, low_x):
        x = high_x + low_x
        left_x = self.local(x)
        right_x = self.global_(x)
        x = left_x + right_x
        weight = self.sigmoid(x)
        x = 2 * weight * high_x + 2 * (1 - weight) * low_x
        return x
    