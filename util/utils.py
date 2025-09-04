import numpy as np
from torch import nn
import torch
import os
import torch
import torch.nn.functional as F
import random
import cv2

# med_frq = [0.382900, 0.452448, 0.637584, 0.377464, 0.585595,
#            0.479574, 0.781544, 0.982534, 1.017466, 0.624581,
#            2.589096, 0.980794, 0.920340, 0.667984, 1.172291,
#            0.862240, 0.921714, 2.154782, 1.187832, 1.178115,
#            1.848545, 1.428922, 2.849658, 0.771605, 1.656668,
#            4.483506, 2.209922, 1.120280, 2.790182, 0.706519,
#            3.994768, 2.220004, 0.972934, 1.481525, 5.342475,
#            0.750738, 4.040773]
# msrs_frq = [0.02571784, 0.29890252, 1.45402557, 1.67359815, 1.89785113, 
#             1.0, 0.65868072, 3.33466205, 0.53685317]
msrs_frq = [0.65, 0.89, 0.96, 1.04, 1.01, 1.02, 1.13, 1.11, 1.07]
# msrs_frq = [0.75, 0.89, 0.96, 1.04, 1.01, 1.02, 1.13, 1.11, 1.07]
# device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# msrs_frq = [0.75, 0.89, 0.96, 1.04, 1.01, 1.02, 1.13, 1.11, 1.07]

# model_urls = {
#     'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
#     # 'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
#     # 'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
#     # 'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
#     # 'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
# }

# label_colours = [(0, 0, 0), # 0=background
#                  (64, 0, 128),  # car
#                  (64, 64, 0),   # person
#                  (0, 128, 192), # bike
#                  (0, 0, 192), # curve
#                  (128, 128, 0), # car_stop
#                  (64, 64, 128), # guardrail
#                  (192, 128, 128), # color_cone
#                  (192, 64, 0)] #bump

class CrossEntropyLoss2d_eval(nn.Module):
    def __init__(self, weight):
        super(CrossEntropyLoss2d_eval, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss(torch.from_numpy(np.array(weight)).float(), size_average=False, reduce=False)

    def forward(self, inputs_scales, targets_scales):
        losses = []
        inputs = inputs_scales
        targets = targets_scales
        # for inputs, targets in zip(inputs_scales, targets_scales):
        # mask = targets > 0
        mask = (targets > 0)
        mask[0] = False
        # targets_m = targets.clone()
        # targets_m[mask] -= 1
        # loss_all = self.ce_loss(inputs, targets_m.long())
        # losses.append(torch.sum(torch.masked_select(loss_all, mask)) / torch.sum(mask.float()))
        if torch.sum(mask.float()) != 0:
            loss_all = self.ce_loss(torch.masked_select(inputs, mask), torch.masked_select(targets, mask)) / (torch.sum(mask.float()))
        else:
            loss_all = torch.tensor(0., requires_grad=True)
        losses.append(loss_all)
        total_loss = sum(losses)
        return total_loss

# class CrossEntropyLoss2d(nn.Module):
#     def __init__(self, weight=med_frq):
#         super(CrossEntropyLoss2d, self).__init__()
#         # self.ce_loss = nn.CrossEntropyLoss(torch.from_numpy(np.array(weight)).float(), size_average=False, reduce=False)
#         self.ce_loss = nn.CrossEntropyLoss(weight, reduction='mean')
#         # self.ce_loss_edge = nn.CrossEntropyLoss(reduction='mean')
        

#     def forward(self, inputs_scales, targets_scales):
#         # losses = []
#         # targets_scales = torch.unsqueeze(targets_scales, 0)
#         # inputs_scales = inputs_scales.permute(1, 0, 2, 3)
#         loss_all = self.ce_loss(inputs_scales, targets_scales)
#         # losses = torch.sum(loss_all) / (inputs_scales.shape[2] * inputs_scales.shape[3])
#         # for inputs, targets in zip(inputs_scales, targets_scales):
#             # 标签为one-hot编码，targets=1所在通道即为该像素点的类别。
#             # mask = (targets == 1)
#             # # 第0个通道代表背景类别，不参与计算损失
#             # mask[0] = False
#             # if torch.sum(mask.float()) != 0:
#             # loss_all = self.ce_loss(torch.masked_select(inputs, mask), torch.masked_select(targets, mask)) / (torch.sum(mask.float()))
#             # else:
#             #     loss_all = torch.tensor(0., requires_grad=True)
#             # loss_all = self.ce_loss(inputs, targets)
#             # losses.append(loss_all)
#             # targets_m = targets.clone()
#             # targets_m[mask] -= 1
#             # values, indices = torch.max(inputs, dim=0)
#             # loss_all = self.ce_loss(inputs, targets)
#             # losses.append(loss_all)
#             # if torch.sum(mask.float()) != 0:
#             #     losses.append(torch.sum(torch.masked_select(loss_all, mask)) / (torch.sum(mask.float()) + 1e-10))
#             # else:
#             #     losses.append(torch.sum(torch.masked_select(loss_all, mask)))
#         # 把所有batch的损失加起来取平均   
#         # total_loss = losses / len(inputs_scales)
#         return loss_all
    
# class BCELoss(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.bceloss = nn.BCELoss(reduction='mean')
    
#     def forward(self, input_scales, target_scales):
#         loss_edge = self.bceloss(input_scales, target_scales)
#         return loss_edge

class TotalLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # self.alpha = nn.Parameter(torch.tensor(0.8, requires_grad=True))
        # self.alpha = nn.Parameter(self.alpha.data.clamp(0.5, 1.0))
        # self.beta = nn.Parameter(torch.tensor(0.1, requires_grad=True))
        # self.beta = nn.Parameter(torch.clamp(self.beta, min=0.0, max=0.5))
        # self.beta = nn.Parameter(torch.tensor(0.2, requires_grad=True))
        # self.weight_edge = nn.Parameter(torch.tensor(weight_edge, requires_grad=True))
        # self.loss_edge = nn.BCELoss(weight_edge, reduction='mean')
        # self.weight = nn.Parameter(weight.to('cuda:0'), requires_grad=True)

        # PST900没有边界损失
        # self.loss_edge = nn.CrossEntropyLoss(reduction='mean')
        self.loss = nn.CrossEntropyLoss(reduction='mean')
    
    def forward(self, input_scales, target_scales):
        # if input_scales_edge.shape[0] == 1:
        #     input_scales_edge = input_scales_edge.squeeze(0)
        #     target_scales_edge = target_scales_edge.squeeze(0)
        # self.beta = nn.functional.softplus(self.beta)
        loss_cate = self.loss(input_scales, target_scales)
        # loss_edge = self.loss_edge(input_scales_edge, target_scales_edge)
        # loss = loss_cate + loss_edge
        return loss_cate


# hxx add, focal loss
class FocalLoss(nn.Module):
    def __init__(self, gamma=0, weight=None, size_average=True):
        # gamma：调节因子，用于调整易分类样本和难分类样本的权重
        # weight：对不同类别进行加权
        # size_average：计算损失时是否对类别进行平均
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.size_average = size_average
        # 用于计算负对数似然损失
        self.loss = nn.NLLLoss(weight=torch.from_numpy(np.array(weight)).float(),
                                 size_average=self.size_average, reduce=False)

    def forward(self, input, target):
        # (1 - F.softmax(input, 1))**2：将预测错误度平方，用于增加难分类样本的权重
        return self.loss((1 - F.softmax(input, 1))**2 * F.log_softmax(input, 1), target)


class FocalLoss2d(nn.Module):
    def __init__(self, weight=msrs_frq, gamma=0):
        super(FocalLoss2d, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.fl_loss = FocalLoss(gamma=self.gamma, weight=self.weight, size_average=False)

    def forward(self, inputs_scales, targets_scales):
        losses = []
        for inputs, targets in zip(inputs_scales, targets_scales):
            mask = targets > 0
            targets_m = targets.clone()
            targets_m[mask] -= 1
            loss_all = self.fl_loss(inputs, targets_m.long())
            losses.append(torch.sum(torch.masked_select(loss_all, mask)) / torch.sum(mask.float()))
        total_loss = sum(losses)
        return total_loss

# 边界框损失计算
# def DiceLoss(target_edge, pred, smooth=1e-5):
#     # target_edge /= 255
#     intersection = 0
#     union = 0
#     for i in range(target_edge.shape[0]):
#         intersection += (target_edge[i] * pred[i]).sum()
#         union += target_edge[i].sum() + pred[i].sum()
#     # dice系数
#     dice_score = (2 * intersection + smooth) / (union + smooth)
#     dice_loss = 1 - dice_score / target_edge.shape[0]
#     return dice_loss

def get_palette():
    unlabelled = [0, 0, 0]
    car = [64,0,128]
    person     = [64,64,0]
    bike       = [0,128,192]
    curve      = [0,0,192]
    car_stop   = [128,128,0]
    guardrail  = [64,64,128]
    color_cone = [192,128,128]
    bump       = [192,64,0]
    palette    = np.array([unlabelled,car, person, bike, curve, car_stop, guardrail, color_cone, bump])
    return palette

def visualize_result(predictions, info, output):
    palette = get_palette()
    # pred_color = color_label_eval(preds)
#     # aggregate images and save
#     im_vis = pred_color.astype(np.uint8)
#     im_vis = im_vis.transpose(2, 1, 0)
#     img_name = str(info)
#     # 保存图像
#     cv2.imwrite(os.path.join(args.output, img_name + '.png'), im_vis)
    img = np.zeros((predictions.shape[0], predictions.shape[1], predictions.shape[2], 3), dtype=np.uint8)
    # for cid in range(0, len(palette)):
    #     for idx in range(predictions.shape[0]):
    #         if (predictions[idx] == cid).sum() == 0:
    #             continue            
    #         img[idx][predictions[idx] == cid] = palette[cid]
    for idx in range(predictions.shape[0]):
        for cid in range(len(palette)):
            if (predictions[idx] == cid).sum() != 0:
                img[idx][predictions[idx] == cid] = palette[cid]
    for i in range(img.shape[0]):
        # img[i] = img[i].astype(np.uint8)
        # aggregate images and save
        # im_vis = pred_color.astype(np.uint8)
        # im_vis = im_vis.transpose(2, 1, 0)
        img_name = str(info) + '_' + str(i)
        # 保存图像
        cv2.imwrite(os.path.join(output, img_name + '.png'),  cv2.cvtColor(img[i], cv2.COLOR_RGB2BGR))

# def color_label_eval(label):
#     # label = label.clone().cpu().data.numpy()
#     # np.vectorize():将一个普通函数提升为一个能作用于数据的函数
#     colored_label = np.vectorize(lambda x: label_colours[int(x)])
#     colored = np.asarray(colored_label(label)).astype(np.float32)
#     colored = colored.squeeze()
#     # return torch.from_numpy(colored.transpose([1, 0, 2, 3]))
#     return colored.transpose([0, 2, 1])

# def color_label(label):
#     label = label.clone().cpu().data.numpy()
#     colored_label = np.vectorize(lambda x: label_colours[int(x)])
#     colored = np.asarray(colored_label(label)).astype(np.float32)
#     colored = colored.squeeze()
#     try:
#         # [batch_size, height, width, channels]->[height, batch_size, width, channels]
#         return torch.from_numpy(colored.transpose([1, 0, 2, 3]))
#     except ValueError:
#         return torch.from_numpy(colored[np.newaxis, ...])


def print_log(global_step, epoch, local_count, count_inter, dataset_size, loss, time_inter):
    print('Step: {:>5} Train Epoch: {:>3} [{:>4}/{:>4} ({:3.1f}%)]    '
          'Loss: {:.6f} [{:.2f}s every {:>4} data]'.format(
        global_step, epoch, local_count, dataset_size,
        100. * local_count / dataset_size, loss.data, time_inter, count_inter))

def save_ckpt(ckpt_dir, model, optimizer, global_step, best_epoch, best_miou):
    # usually this happens only on the start of a epoch
    # epoch_float = epoch + (local_count / num_train)
    state = {
        'global_step': global_step,
        'best_miou':best_miou,
        'epoch': best_epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    path = os.path.join(ckpt_dir, 'epoch%d_%.3f.pth' % (best_epoch,best_miou))
    torch.save(state, path)


def load_ckpt(model, optimizer, model_file, device):
    if os.path.isfile(model_file):
        if device.type == 'cuda':
            checkpoint = torch.load(model_file)
        else:
            # lambda storage, loc: storage 接收两个参数storage（实际存储）和loc（原始存储），返回storage
            checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
        # checkpoint = torch.load(model_file)
        model.load_state_dict(checkpoint['state_dict'])
        if optimizer:
            optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(model_file, checkpoint['epoch']))
        # step = checkpoint['global_step']
        # epoch = checkpoint['epoch']
        # return step, epoch
    else:
        print("=> no checkpoint found at '{}'".format(model_file))
        os._exit(0)

# added by hxx for iou calculation
# def intersectionAndUnion(imPred, imLab, numClass):
#     # imPred += 1 # hxx
#     # imLab += 1 # label 应该是不用加的
#     # Remove classes from unlabeled pixels in gt image.
#     # We should not penalize detections in unlabeled portions of the image.
#     imPred = np.asarray(imPred).copy()
#     imLab = np.asarray(imLab).copy()
#     # 对每一行计算混淆矩阵
#     # for single_image, single_label in zip(imPred, imLab):
#     #     for row_image, row_label in zip(single_image, single_label):
#     #         mask = (row_label >= 0) & (row_label < numClass)
#     #         hist = np.bincount(numClass * row_label[mask].astype(float) + row_image[mask], minlength=numClass ** 2).reshape(numClass)
#      # 将预测与标签一致的像素保留(算总体数据集的mIou时不包含背景类别，单独算各个类别的iou时包含背景类别)
#     imPred = imPred * (imLab > 0)

#     # Compute area intersection:
#     # 计算intersection的直方图
#     # area_intersection:每个区间的个数。一个区间代表一个类，即每个类包含的个数
#     # _:接收的是直方图的边缘值
#     intersection = imPred * (imPred == imLab)
#     (area_intersection, _) = np.histogram(intersection, bins=numClass, range=(1, numClass))

#     # Compute area union:
#     (area_pred, _) = np.histogram(imPred, bins=numClass, range=(1, numClass))
#     (area_lab, _) = np.histogram(imLab, bins=numClass, range=(1, numClass))
#     area_union = area_pred + area_lab - area_intersection

#     # imPred += 1 # hxx
#     # imLab += 1 # label 应该是不用加的
#     # imPred = imPred * (imLab > 0)
#     # Compute area intersection:
#     # index = index * (index > 0)
#     # 过滤掉标签为0即背景或非目标类别预测结果.
#     # intersection = imPred[np.where(imPred == imLab)[0]]

#     return (area_intersection, area_union)

# def accuracy(preds, label):
#     # 忽略背景类别
#     valid = (label > 0)
#     acc_sum = (valid * (preds == label)).sum()
#     valid_sum = valid.sum()
#     # 1e-10可以避免除以0的情况发生
#     acc = float(acc_sum) / (valid_sum + 1e-10)
#     # # 算背景类别
#     # acc_sum = (preds == label).sum()
#     # acc = float(acc_sum) / (480 * 640)
#     # return acc, 480 * 640
#     return acc, valid_sum

# def macc(preds, label, num_class):
#     a = np.zeros(num_class)
#     b = np.zeros(num_class)
#     # 忽略背景类别
#     for i in range(1, num_class):
#         mask = (label == i)
#         # 计算第i个类别预测中有多少是正确的
#         a_sum = (mask * preds == i).sum()
#         # 计算真实标签中类别为i的个数
#         b_sum = mask.sum()
#         a[i] = a_sum
#         b[i] = b_sum
#     return a, b

# def calculate_result(cf):
#     n_class = cf.shape[0]
#     conf = np.zeros((n_class,n_class))
#     IoU = np.zeros(n_class)
#     conf[:,0] = cf[:,0]/cf[:,0].sum()
#     for cid in range(1,n_class):
#         if cf[:,cid].sum() > 0:
#             conf[:,cid] = cf[:,cid]/cf[:,cid].sum()
#             IoU[cid]  = cf[cid,cid]/(cf[cid,1:].sum()+cf[1:,cid].sum()-cf[cid,cid])
#     overall_acc = np.diag(cf[1:,1:]).sum()/cf[1:,:].sum()
#     acc = np.diag(conf)
#     return overall_acc, acc, IoU

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, num_classes, ignore_label, device):
        self.num_classes = num_classes
        self.ignore_label = ignore_label
        self.hist = torch.zeros(num_classes, num_classes).to(device)
        # self.initialized = False
        # # pre_acc:当前样本中的acc
        # self.pre_acc = None
        # self.acc = None
        # self.sum_acc = None
        # # 样本数量
        # self.count = None

    # def initialize(self, acc, valid):
    #     self.pre_acc = acc
    #     self.acc = acc
    #     self.sum_acc = acc * valid
    #     self.count = valid
    #     self.initialized = True

    def update(self, output, target):
        keep = target != self.ignore_label
        self.hist += torch.bincount(target[keep] * self.num_classes + output[keep], minlength=self.num_classes ** 2).view(self.num_classes, self.num_classes)
        # if not self.initialized:
        #     self.initialize(acc, valid)
        # else:
        #     self.add(acc, valid)
    def compute_iou(self):
        # self.hist.sum(0):按列求和，表示每个类别的预测为该类别的总数
        # self.hist.sum(1):按行求和，真实为该类别的样本总数
        ious = self.hist.diag() / (self.hist.sum(0) + self.hist.sum(1) - self.hist.diag())
        ious[ious.isnan()] = 0.
        miou = ious.mean().item()
        ious *= 100
        miou *= 100
        # round():四舍五入
        return ious.cpu().numpy().round(2).tolist(), round(miou, 2)
    
    def compute_pixel_acc(self):
        acc = self.hist.diag() / self.hist.sum(1)
        acc[acc.isnan()] = 0.
        macc = acc.mean().item()
        acc *= 100
        macc *= 100
        return acc.cpu().numpy().round(2).tolist(), round(macc, 2)

    # def add(self, acc, valid):
    #     self.pre_acc = acc
    #     self.sum_acc += acc * valid
    #     self.count += valid
    #     self.avg = self.sum_acc / self.count

    def value(self):
        return self.pre_acc

    def average(self):
        return self.avg
    
def set_random_seed(seed, deterministic=False):
    """Set random seed."""
    # https://blog.csdn.net/qq_41118968/article/details/118937199
    # https://wenku.baidu.com/view/ac0a1720753231126edb6f1aff00bed5b9f373bd.html
    random.seed(seed)
    np.random.seed(seed)
    # 设置PyTorch的CPU随机数生成器的种子
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        # 设置PyTorch的当前GPU随机数生成器的种子
        torch.cuda.manual_seed(seed)
        # 为所有GPU设置相同的随机种子
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            # 对于一个PyTorch 的函数接口，没有确定性算法实现，只有非确定性算法实现，同时设置了use_deterministic_algorithms()，那么会导致运行时错误。
            # torch.use_deterministic_algorithms(True) # >=1.8  
            # torch.set_deterministic(True) # before 1.8  
            # RuntimeError: SpatialClassNLLCriterion_updateOutput does not have a deterministic implementation, but you set 'torch.set_deterministic(True)'
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        else:
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True