import argparse
import torch
import torchvision
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
import sys
sys.path.append('/workspace/try1')
from model.builder import EncoderDecoder as seg
import torch.optim
import dataset.msrs as data
from util.utils import load_ckpt, AverageMeter, visualize_result
import torchvision.transforms.functional as TF
import math
import util.utils as utils
# from util.utils import load_ckpt, color_label_eval, CrossEntropyLoss2d_eval, calculate_result
# from dataset.msrs import scaleNorm

parser = argparse.ArgumentParser(description='RGBT Sementic Segmentation')
parser.add_argument('--data-dir', default='/workspace/try1/MFNet/test', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-o', '--output', default='/workspace/try1/eval_result', metavar='DIR',
                    help='path to output')
parser.add_argument('--cuda', action='store_true', default=True,
                    help='enables CUDA training')
parser.add_argument('--latest-ckpt', default='/workspace/try1/checkpoint_save/weight_baseline.pth', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--num-class', default=9, type=int,
                    help='number of classes')
parser.add_argument('--visualize', default=True, action='store_true',
                    help='if output image')
parser.add_argument('--pre', default='', type=str, metavar='PATH',
                    help='path to pretrained model (default: none)')

args = parser.parse_args()
device = torch.device("cuda:0" if args.cuda and torch.cuda.is_available() else "cpu")
image_w = 640
image_h = 480
# img_mean=[0.485, 0.456, 0.406]
# img_std=[0.229, 0.224, 0.225]

# transform
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        image, thermal, label, label_edge = sample['image'], sample['thermal'], sample['label'], sample['label_edge']
        # image, thermal, label = sample['image'], sample['thermal'], sample['label']
        image = image.transpose((2, 0, 1))
        thermal = thermal.transpose((2, 0, 1)).astype(np.float64)
        return {'image': torch.from_numpy(image).float(), 'thermal': torch.from_numpy(thermal).float(), 'label': torch.from_numpy(label).float(), 'label_edge': torch.from_numpy(label_edge).float()}

# class Resize:
#     def __init__(self, size) -> None:
#         """Resize the input image to the given size.
#         Args:
#             size: Desired output size. 
#                 If size is a sequence, the output size will be matched to this. 
#                 If size is an int, the smaller edge of the image will be matched to this number maintaining the aspect ratio.
#         """
#         self.size = size

#     def __call__(self, sample:list) -> list:
#         H, W = sample['image'].shape[1:]
#         # image, thermal = sample['image'], sample['thermal']
#         # image = torchvision.transforms.functional.to_pil_image(image)
#         # image.save('/workspace/try1/MFNet/mid/crop_pre.png')
#         # thermal = torchvision.transforms.functional.to_pil_image(thermal)
#         # thermal.save('/workspace/try1/MFNet/mid/crop_pre_the.png')

#         # scale the image 
#         scale_factor = self.size[0] / min(H, W)
#         nH, nW = round(H*scale_factor), round(W*scale_factor)
#         for k, v in sample.items():
#             # if k == 'image': 
#             #     sample[k] = TF.resize(v, (nH, nW), TF.InterpolationMode.BILINEAR)
#             # else:
#             #     if k == 'label' or k == 'label_edge':
#             #         v = v.reshape(1, H, W)               
#             #         sample[k] = TF.resize(v, (nH, nW), TF.InterpolationMode.NEAREST)
#             #         sample[k] = sample[k].reshape(nH, nW)
#             #     else:
#             #         sample[k] = TF.resize(v, (nH, nW), TF.InterpolationMode.NEAREST)
#             if k == 'label' or k == 'label_edge':     
#                 v = v.reshape(1, H, W)           
#                 sample[k] = TF.resize(v, (nH, nW), TF.InterpolationMode.NEAREST)
#                 sample[k] = sample[k].reshape(nH, nW)
#             else:
#                 sample[k] = TF.resize(v, (nH, nW), TF.InterpolationMode.BILINEAR)
#         # img = TF.resize(img, (nH, nW), TF.InterpolationMode.BILINEAR)
#         # mask = TF.resize(mask, (nH, nW), TF.InterpolationMode.NEAREST)

#         # make the image divisible by stride
#         alignH, alignW = int(math.ceil(nH / 32)) * 32, int(math.ceil(nW / 32)) * 32
        
#         for k, v in sample.items():
#             if k == 'label' or k == 'label_edge':     
#                 v = v.reshape(1, H, W)           
#                 sample[k] = TF.resize(v, (alignH, alignW), TF.InterpolationMode.NEAREST)
#                 sample[k] = sample[k].reshape(alignH, alignW)
#             else:
#                 sample[k] = TF.resize(v, (alignH, alignW), TF.InterpolationMode.BILINEAR)
#             # if k == 'image': 
#             #     sample[k] = TF.resize(v, (alignH, alignW), TF.InterpolationMode.BILINEAR)
#             # else:
#             #     if k == 'label' or k == 'label_edge':
#             #         v = v.reshape(1, nH, nW)               
#             #         sample[k] = TF.resize(v, (alignH, alignW), TF.InterpolationMode.NEAREST)
#             #         sample[k] = sample[k].reshape(alignH, alignW)
#             #     else:
#             #         sample[k] = TF.resize(v, (alignH, alignW), TF.InterpolationMode.NEAREST)
#         # image = sample['image']
#         # thermal = sample['thermal']
#         # image = torchvision.transforms.functional.to_pil_image(image)
#         # image.save('/workspace/try1/MFNet/mid/crop_rgb.png')
#         # thermal = torchvision.transforms.functional.to_pil_image(thermal)
#         # thermal.save('/workspace/try1/MFNet/mid/crop_the.png')
#         return sample

class Normalize(object):
    def __call__(self, sample):
        image, thermal = sample['image'], sample['thermal']
        # image_ = torchvision.transforms.functional.to_pil_image(image)
        # image_.save('/workspace/try1/MFNet/mid/norm_pre.png')
        # thermal_ = torchvision.transforms.functional.to_pil_image(thermal)
        # thermal_.save('/workspace/try1/MFNet/mid/norm_pre_the.png')
        # origin_image = image.clone()
        # origin_thermal = thermal.clone()
        # image = torchvision.transforms.Normalize(mean=[0.245, 0.275, 0.236], std=[0.235, 0.235, 0.239])(image)
        # image = torchvision.transforms.Normalize(mean=[0.222, 0.259, 0.213], std=[0.249, 0.284, 0.289])(image)
        # image = torchvision.transforms.Normalize(mean=[0.232, 0.266, 0.231], std=[0.231, 0.233, 0.237])(image)
        image = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image)
        # image = torchvision.transforms.Normalize(mean=[0.221, 0.259, 0.230], std=[0.228, 0.231, 0.236])(image)
        # thermal = torchvision.transforms.Normalize(mean=[0.395, 0.395, 0.395], std=[0.184, 0.184, 0.184])(thermal)
        thermal = torchvision.transforms.Normalize(mean=[0.430, 0.430, 0.430], std=[0.226, 0.226, 0.226])(thermal)
        # thermal = torchvision.transforms.Normalize(mean=[0.430, 0.430, 0.430], std=[0.123, 0.123, 0.123])(thermal)
        # image_ = torchvision.transforms.functional.to_pil_image(image)
        # image_.save('/workspace/try1/MFNet/mid/norm_rgb.png')
        # thermal_ = torchvision.transforms.functional.to_pil_image(thermal)
        # thermal_.save('/workspace/try1/MFNet/mid/norm_the.png')
        # sample['origin_image'] = origin_image
        # sample['origin_thermal'] = origin_thermal
        sample['image'] = image
        sample['thermal'] = thermal
        return sample
    
# def visualize_result(img, thermal, preds, info, args):
#     # segmentation
#     # 移除第0维，若第0维不是1则不会生效
#     # img = img.squeeze(0).transpose(0, 2, 1)
#     the = thermal.squeeze(0).transpose(2, 1, 0)
#     # 将the的数值范围缩放到[0, 255]，/the.max()这里max是原始the数组的max即*255之前的the数组，原始the数组的数值范围是[0, 1]./the.max()是为了防止溢出
#     the = (the * 255 / the.max()).astype(np.uint8)
#     # 颜色映射
#     the = cv2.applyColorMap(the, cv2.COLORMAP_JET)
#     # C, H, W -> H, W, C
#     # the = the.transpose(2,1,0)
#     # prediction
#     pred_color = color_label_eval(preds)
#     # aggregate images and save
#     im_vis = pred_color.astype(np.uint8)
#     im_vis = im_vis.transpose(2, 1, 0)
#     img_name = str(info)
#     # 保存图像
#     cv2.imwrite(os.path.join(args.output, img_name + '.png'), im_vis)

def inference(args, model=None, g=None):
    # cf = np.zeros((9, 9))
    flag = False
    if model is None:
        model = seg(cfg=args, num_class=9, criterion=None, norm_layer=nn.BatchNorm2d)
        load_ckpt(model, None, args.latest_ckpt, device)
        flag = True
    model.eval()
    model.to(device)
    # val_data = data.MSRS(transform=torchvision.transforms.Compose([scaleNorm(), ToTensor(), Normalize()]), phase_train=False, data_dir=args.data_dir)
    val_data = data.MSRS(transform=torchvision.transforms.Compose([ToTensor(), Normalize()]), phase_train=False, data_dir=args.data_dir)
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=1, pin_memory=False, generator=g)
    metrics = AverageMeter(9, 255, device)
    weighted = utils.TotalLoss()
    losses=0
    # acc_meter = AverageMeter()
    # intersection_meter = AverageMeter()
    # union_meter = AverageMeter()
    # a_meter = AverageMeter()
    # b_meter = AverageMeter()
    # loss_meter = AverageMeter()
    # cel_loss = CrossEntropyLoss2d_eval()
    with torch.no_grad():
        for batch_idx, sample in enumerate(val_loader):
            #todo batch=1，这里要查看sample的size，决定怎么填装image thermal label，估计要用到for循环
            # origin_image = sample['origin_image'].numpy()
            # origin_thermal = sample['origin_thermal'].numpy()
            image = sample['image'].to(device)
            thermal = sample['thermal'].to(device)
            # label = sample['label'].numpy()
            label = sample['label'].to(device).long()
            label_edge = sample['label_edge'].to(device).long()
            # label = torch.from_numpy(label).long()
            # label_edge = sample['label_edge'].numpy()
            # label = sample['label'].to(device)
            # bc, h, w = label.shape

            # image = image.cuda()
            # thermal = thermal.cuda()

            # conv = nn.Conv2d(64, args.num_class, kernel_size=1)
            # one-hot编码
            # target = torch.zeros(bc, args.num_class, h, w)
            # for c in range(args.num_class):
            #     target[:, c, :, :][label == c] = 1
            # target = target.to(device)
            # with torch.no_grad():
            pred, pred_edge = model(image, thermal)
                # pred = model(image)
                # pred_out = pred['out'].softmax(dim=1)
                # pred = conv(pred)
                # pred = nn.functional.interpolate(pred, (480, 640))
            
            # predictions = pred.argmax(1)
            # for gtcid in range(9): 
            #     for pcid in range(9):
            #         gt_mask      = label == gtcid
            #         # gt_mask = torch.from_numpy(gt_mask)
            #         pred_mask    = predictions == pcid
            #         # pred_mask = torch.from_numpy(pred_mask)
            #         intersection = gt_mask * pred_mask
            #         cf[gtcid, pcid] += int(intersection.sum())
            

    # overall_acc, acc, IoU = calculate_result(cf)

    # print('| overall accuracy:', overall_acc)
    # print('| accuracy of each class:', acc)
    # print('| mACC:', acc.mean())
    # print('| IoU:', IoU)
    # print('| mIou:', IoU.mean())
            # print(pred.shape)
            # 沿着维度1（类别维度）寻找最大值。
            # torch.max()返回一个元组，元组中第一个数为最大值，第二个数为最大值索引，[1]用来返回最大值索引
            # macc、miou忽略背景类别，output是每个像素点预测的类别.这里是包含背景类别的，所以要在计算这俩指标的函数里面加上判断是否为背景类别然后进行忽略
            output = pred.argmax(dim=1)
            bc, h, w = label.shape
            target = torch.zeros(bc, args.num_class, h, w)
            for c in range(args.num_class):
                target[:, c, :, :][label == c] = 1
            target = target.to(device)
            target_edge = torch.zeros(bc, 2, h, w)
            target_edge[:, 0, :, :][label_edge == 0] = 1
            target_edge[:, 1, :, :][label_edge == 255] = 1
            target_edge = target_edge.to(device)
            losses += weighted(pred, target, pred_edge, target_edge)
            # output = torch.max(pred, 1)[1]
            # output = output.cpu()
            metrics.update(output, label)
            # label = label.squeeze(0)
            # label_edge = label_edge.squeeze(0)
            # loss = cel_loss(pred, target)
            # loss_meter.update(loss)
            # valid是标签>0的个数
            # acc, valid = accuracy(output, label)
            # acc_meter.update(acc, valid)
            # intersection, union = intersectionAndUnion(output, label, args.num_class)
            # intersection_meter.update(intersection)
            # union_meter.update(union)
            # # a_m: 预测中准确的个数
            # # b_m: 某个类别真实标签的个数
            # a_m, b_m = macc(output, label, args.num_class)
            # a_meter.update(a_m)
            # b_meter.update(b_m)
            # print('[{}] iter {}, accuracy: {}'
            #       .format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            #               batch_idx, acc))

            # img = image.cpu().numpy()
            # print('origin iamge: ', type(origin_image))

    loss = losses / len(val_loader)
    ious, miou = metrics.compute_iou()
    acc, macc = metrics.compute_pixel_acc()
    # mIou = (intersection_meter.sum_acc / (union_meter.sum_acc + 1e-10)).mean()
    # # mIou = 0
    # # for i in range(len(iou)):
    # #     mIou += iou[i]
    # # # 不包含背景类别
    # # mIou = mIou / 8
    # # return iou.mean()
    # acc_cate = a_meter.average() / (b_meter.average() + 1e-10)
    # mAcc = (a_meter.average() / (b_meter.average() + 1e-10)).mean()
    # acc_cate = a_meter.average() / (b_meter.average() + 1e-10)
    # mAcc = 0
    # for i in range(len(acc_cate)):
    #     mAcc += acc_cate[i]
    # mAcc = mAcc / 8
    if flag:
        for i, _iou in enumerate(ious):
            print('class [{}], IoU: {:.1f}'.format(i, _iou))
        for i, _acc_cate in enumerate(acc):
            print('class [{}], ACC: {:.1f}'.format(i, _acc_cate))
    # print('mAcc:',mAcc.mean())
    # print('[Eval Summary]:')
    # print('Mean IoU: {:.4}, Accuracy: {:.2f}%'
    #       .format(iou.mean(), acc_meter.average() * 100))
    # return overall_acc, acc, IoU
    return ious, miou, acc, macc, loss

if __name__ == '__main__':
    # if not os.path.exists(args.output):
    #     os.mkdir(args.output)
    ious, miou, acc, macc, loss = inference(args)
    print('Mean IoU: {:.1f}'.format(miou))
    print('Mean Accuracy: {:.1f}'.format(macc))
