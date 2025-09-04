from torch import nn
import torch.utils.model_zoo as model_zoo
import torch

model_urls = {
    # 'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    # 'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    # 'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    # 'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    # 'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'segformer': '/workspace/experiment2/pretrained/mit_b2.pth'
}

class pretrained(nn.Module):
    def __init__(self):
        pretrain_dict = torch.load(model_urls['segformer'])
        model_dict = {}
        # 获取当前模型的状态字典
        for k, v in pretrain_dict.items():
            print('key:{}'.format(k))
            print(v.shape)
        # print(pretrain_dict)

if __name__ == '__main__':
    pretrained()