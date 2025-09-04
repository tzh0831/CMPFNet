from torch import nn
import torch
from model.encoder import Backbone
import torch.nn.functional as F
    
class ConvBnLeakyRelu2d(nn.Module):
    # convolution
    # batch normalization
    # leaky relu
    def __init__(self, in_channels, out_channels, kernel_size=1, padding=0, stride=1, dilation=1, groups=1):
        super(ConvBnLeakyRelu2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups)
        self.bn   = nn.BatchNorm2d(out_channels)
    def forward(self, x):
        return F.leaky_relu(self.bn(self.conv(x)), negative_slope=0.2)

class Decoder(nn.Module):
    def __init__(self, in_channels=[64, 128, 256, 512, 512], num_class=9, embed_dim=768):
        super().__init__()
        self.decode5 = ConvBnLeakyRelu2d(1024, 512)
        self.decode4     = ConvBnLeakyRelu2d(768, 256)
        self.decode3     = ConvBnLeakyRelu2d(384, 128)
        self.decode2     = ConvBnLeakyRelu2d(192, 64)
        # self.decode1     = ConvBnLeakyRelu2d(128, 64)
        self.decode = nn.Conv2d(64*2, num_class, kernel_size=1)
        self.sig = nn.Sigmoid()
        
        self.outconv_edge = nn.Conv2d(64, 2, kernel_size=3, stride=1, padding=1, bias=False)
        # self.edge_up = nn.Conv2d(2, 2, kernel_size=3, stride=1, padding=1)
        self.backbone = Backbone()
        self.dropout = nn.Dropout2d(0.1)
        # self.fea = FeatureAugment(pool=(60, 80), dim=64)
        # self.soft = nn.Softmax(dim=1)

    def forward(self, x, edge):
        x1, x2, x3, x4, x5 = x
        x5 = x5.reshape(x5.shape[0], 15, 20, 512).permute(0, 3, 1, 2)
        x4 = x4.reshape(x4.shape[0], 15, 20, 512).permute(0, 3, 1, 2)
        x = self.decode5(torch.concat((x5, x4), dim=1))
        x = nn.functional.interpolate(x, (30, 40), mode='nearest') # unpool4
        x3 = x3.reshape(x3.shape[0], 30, 40, 256).permute(0, 3, 1, 2)
        x = self.decode4(torch.concat((x, x3), dim=1))
        x = nn.functional.interpolate(x, (60, 80), mode='nearest') # unpool3
        x2 = x2.reshape(x2.shape[0], 60, 80, 128).permute(0, 3, 1, 2)
        x = self.decode3(torch.concat((x, x2), dim=1))
        x = nn.functional.interpolate(x, (120, 160), mode='nearest') # unpool2
        x1 = x1.reshape(x1.shape[0], 120, 160, 64).permute(0, 3, 1, 2)
        x = self.decode2(torch.concat((x, x1), dim=1))
        x = self.decode(torch.concat((x, edge), dim=1))
        x = nn.functional.interpolate(x, (480, 640), mode='nearest') # unpool1
        x_edge = self.dropout(x)
        edge = self.outconv_edge(edge)
        edge = nn.functional.interpolate(edge, (480, 640), mode='nearest')
        edge = self.dropout(edge)
        return x_edge, edge
