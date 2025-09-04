import torch.nn as nn
import torch.nn.functional as F
import sys
# 添加项目的根目录环境，解决找不到util包的问题
sys.path.append('/workspace/try1')
from util.init_func import init_weight
from util.logger import get_logger
from model.encoder import Backbone
from model.decoder import Decoder

logger = get_logger()

class EncoderDecoder(nn.Module):
    # reduction='mean':所有样本的损失加起来取平均
    def __init__(self, num_class, cfg=None, criterion=nn.CrossEntropyLoss(reduction='mean'), norm_layer=nn.BatchNorm2d, aux_rate=0.1):
        super().__init__()
        self.channels = [64, 128, 320, 512]
        self.norm_layer = norm_layer
        logger.info('Using backbone: ResNet+SegFormer')
        self.backbone = Backbone()
        # 辅助头
        self.aux_head = None
        self.aux_rate = aux_rate
        self.decoder = Decoder(num_class=num_class)
        # self.init_weights(cfg, pretrained=cfg.pre)
        self.criterion = criterion
    
    def init_weights(self, pretrained=None):
        if pretrained:
            logger.info('Loading pretrained model: {}'.format(pretrained))
            self.backbone.init_weights(pretrained=pretrained)
        logger.info('Initing weights ...')
        # kaiming_normal_:正态分布初始化.通常适用于深度卷积网络，特别是使用ReLU激活函数时
        init_weight(self.decoder, nn.init.kaiming_normal_, self.norm_layer, 1e-5, 0.1, mode='fan_in', nonlinearity='relu')
        if self.aux_head:
            init_weight(self.aux_head, nn.init.kaiming_normal_, self.norm_layer, 1e-5, 0.1, mode='fan_in', nonlinearity='relu')
    
    def encode_decode(self, rgb_x, the_x):
        """
        Encode images with backbone and decode into a semantic segmentation map of the same size as input.
        """
        orisize = rgb_x.shape
        x, edge = self.backbone(rgb_x, the_x)
        # out = self.decode_head.forward(x)
        if self.training:
            out, edge = self.decoder.forward(x, edge)
            # out, edge = self.decoder.forward(x, edge)
            #480,640||240,320||120,160||60,80
            # out = F.interpolate(out, size=(orisize[2], orisize[3]), mode='bilinear', align_corners=False)
            # out2 = F.interpolate(out2, size=(orisize[2]// 2, orisize[3]// 2), mode='bilinear', align_corners=False)
            # out3 = F.interpolate(out3, size=(orisize[2]// 4, orisize[3]// 4), mode='bilinear', align_corners=False)
            # out4 = F.interpolate(out4, size=(orisize[2]// 8, orisize[3]// 8), mode='bilinear', align_corners=False)
            # out5 = F.interpolate(out5, size=(orisize[2]// 16, orisize[3]// 16), mode='bilinear', align_corners=False)
            if self.aux_head:
                aux_fm = self.aux_head(x[self.aux_index])
                aux_fm = F.interpolate(aux_fm, size=orisize[2:], mode='bilinear', align_corners=False)
                # return out, aux_fm
                return out, edge, aux_fm
            # return out
            return out, edge
        else:
            out, edge = self.decoder.forward(x, edge)
            # out, edge = self.decoder.forward(x, edge)
            # out = F.interpolate(out, size=(orisize[2], orisize[3]), mode='bilinear', align_corners=False)
            # return out
            return out, edge

    def forward(self, rgb, modal_x, label=None):
        if self.aux_head:
            out, edge, aux_fm = self.encode_decode(rgb, modal_x)
            # out, aux_fm = self.encode_decode(rgb, modal_x)
        else:
            # out = self.encode_decode(rgb, modal_x)
            out, edge = self.encode_decode(rgb, modal_x)
        if label is not None:
            loss = self.criterion(out, label.long())
            if self.aux_head:
                loss += self.aux_rate * self.criterion(aux_fm, label.long())
            return loss
        # return out
        return out, edge
