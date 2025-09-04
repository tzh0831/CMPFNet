import argparse
import os
import torch
import logging
from torch.utils.data import DataLoader
import torch.optim
import torchvision.transforms as transforms
from torchvision.utils import make_grid
from torch import nn
from tensorboardX import SummaryWriter
from thop import profile
import sys
sys.path.append('/workspace/try1')
from model.builder import EncoderDecoder as seg
import dataset.msrs as data
import util.utils as utils
from util.utils import save_ckpt, load_ckpt, set_random_seed, visualize_result, AverageMeter
from torch.optim.lr_scheduler import LambdaLR
from dataset.eval_mfnet import inference
import numpy as np
from util.lr_schedulers import get_scheduler
# import numpy as np
# import matplotlib.pyplot as plt
# from model.MFNet import MFNet as seg



os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# msrs_frq = [0.02571784, 0.29890252, 1.45402557, 1.67359815, 1.89785113, 
#             1.0, 0.65868072, 3.33466205, 0.53685317]
# class_weight = np.array([0.02571784, 0.4, 1.45402557, 1.7, 1.89785113, 
#             1.2, 0.7, 3.33466205, 0.7])
# edge_frq = [0.0, 3.0]
# val_msrs_frq = [0.74, 0.90, 0.99, 0.94, 1.02, 1.05, 1.22, 1.07, 1.04]
# class_weight = np.array([0.75, 0.89, 0.96, 1.04, 1.01, 1.02, 1.13, 1.11, 1.07])
# class_weight = np.array([1.5105, 16.6591, 29.4238, 34.6315, 40.0845, 41.4357, 47.9794, 45.3725, 44.9000])
# class_weight = np.array([1.5105, 15.8163, 30.0804, 41.7054, 38.3325, 39.7838, 46.9887, 46.4685, 44.2373])
# bound_weight = np.array([1.4459, 23.7228])
# bound_weight = np.array([1.4458, 50.4983])
# bound_weight = np.array([1.4458, 10.4983])

parser = argparse.ArgumentParser(description='RGBT Sementic Segmentation')
parser.add_argument('--data-dir', default='/workspace/try1/MFNet/train', metavar='DIR',
                    help='path to MFNet')
parser.add_argument('--cuda', action='store_true', default=True,
                    help='enables CUDA training')
parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run (default: 50)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=4, type=int,
                    metavar='N', help='mini-batch size (default: 10)')

parser.add_argument('--lr', '--learning-rate', default=4e-5, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--weight-decay', '--wd', default=4e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--num_class', default=9, type=int, metavar='N',
                    help='Number of categories in the dataset(default: 8)')
parser.add_argument('--output', default="/workspace/try1/output_pic_train")
parser.add_argument('--output_val', default="/workspace/try1/output_pic_val")

parser.add_argument('--print-freq', '-p', default=5, type=int,
                    metavar='N', help='print batch frequency (default: 50)')
parser.add_argument('--save-epoch-freq', '-s', default=5, type=int,
                    metavar='N', help='save epoch frequency (default: 5)')
parser.add_argument('--latest-ckpt', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--visualize', default=False, action='store_true',
                    help='if output image')
parser.add_argument('--pre', default='/workspace/try1/pretrained/segformer_b4_weights.pth', type=str, metavar='PATH',
                    help='path to pretrained model (default: none)')
parser.add_argument('--lr-decay-rate', default=0.5, type=float,
                    help='decay rate of learning rate (default: 0.8)')
parser.add_argument('--lr-epoch-per-decay', default=5, type=int,
                    help='epoch of per decay of learning rate (default: 50)')
parser.add_argument('--ckpt-dir', default='/workspace/try1/checkpoint_save', metavar='DIR',
                    help='path to save checkpoints')
parser.add_argument('--summary-dir', default='/summary', metavar='DIR',
                    help='path to save summary')
parser.add_argument('--checkpoint', action='store_true', default=False,
                    help='Using Pytorch checkpoint or not')
parser.add_argument("--seed", type=int, default=3305)

args = parser.parse_args()
device = torch.device("cuda:0" if args.cuda and torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
image_w = 640
image_h = 480
# torch.cuda.empty_cache()
# torch.backends.cudnn.enabled = False
def count_params(model):
    # p.numel()获取参数中的元素总数。一个参数的元素总数是指这个参数的矩阵中有几个元素，比如3*3的矩阵，元素个数即为9.
    # param_num: 模型中所有参数的元素总数
    param_num = sum(p.numel() for p in model.eval().parameters())
    tensor = torch.randn(1, 3, 480, 640)
    tensor_ = torch.randn(1, 3, 480, 640)
    flops, params = profile(model, inputs=(tensor, tensor_))
    # /1e6即1000000，转换为以百万为单位
    # return param_num / 1e6
    return flops, params

logs = set()


def init_log(name, fname, level=logging.INFO):
    # logging.INFO: 常规信息输出
    if (name, level) in logs:
        return
    logs.add((name, level))
    # logger: 日志处理器，负责收集日志，并根据设置的level决定是否处理该信息，并将记录的日志信息传递给处理器
    logger = logging.getLogger(name)#实例化logging对象
    logger.setLevel(level)
    # ch: 用于将日志信息输出到流（如标准输出或文件）
    ch = logging.StreamHandler()#处理器 将日志消息输出到流的处理器
    ch.setLevel(level)
    # SLURM: 一个作业调度系统，用于标识任务中的进程
    if "SLURM_PROCID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        # 为日志记录器添加一个过滤器，rank=0时日志消息才会被记录。通常用于多进程环境中，只让主进程输出日志
        logger.addFilter(lambda record: rank == 0)
    else:
        rank = 0
    # 定义日志格式
    format_str = "[%(asctime)s][%(levelname)8s] %(message)s"
    # 创建日志格式器
    formatter = logging.Formatter(format_str)
    # 创建文件处理器，将日志消息写入文件
    fh = logging.FileHandler(filename=fname,encoding='utf-8',mode='a')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    ch.setFormatter(formatter)#设置处理器的格式
    logger.addHandler(ch)#增加处理器
    return logger

def train():
    logger = init_log('global', args.ckpt_dir+'/log.txt',logging.INFO) # 正常命令行的输出也记录
    logger.propagate = 0  #关闭反馈机制 不需要下级给上级反馈输出
    # logger.info('{}\n'.format(pprint.pformat(cfg)))#pprint.pformat 将文件中的信息以字符串的格式打印
    if args.seed is not None:
        set_random_seed(args.seed, deterministic=True)
    # 随机数生成器
    g = torch.Generator()
    g.manual_seed(args.seed)

    # train_data = data.MSRS(transform=transforms.Compose([data.scaleNorm(), data.RandomScale((1.0, 1.4)), data.RandomHSV((0.9, 1.1), (0.9, 1.1), (25, 25)),
    #                                                      data.RandomCrop(image_h, image_w), data.RandomFlip(), data.ToTensor(), data.Normalize()]), phase_train=True, data_dir=args.data_dir)
    
    train_data = data.MSRS(transform=transforms.Compose([data.RandomFlip(), data.ToTensor(), data.RandomHSV(), data.RandomResizedCrop((480, 640), (0.5, 2.0)), data.Normalize()]), phase_train=True, data_dir=args.data_dir)
    

    # train_data = data.MSRS(transform=transforms.Compose([ data.RandomHSV(p=0.5),
    #                                                      data.RandomCrop(image_h, image_w), data.RandomFlip(p=0.5), data.ToTensor(), data.Normalize()]), phase_train=True, data_dir=args.data_dir)

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=False, generator=g)
    metrics = AverageMeter(9, 255, device)
    # os.environ['PYTORCH_CUDA_ALLOC_CONF'] = '1'
    # os.environ['PYTORCH_CUDA_ALLOC_CONF'] += ':max_split_size_mb=46'
    # print(torch.cuda.memory_allocated())
    # num_train = len(train_data)

    # model = torch.hub.load('pytorch/vision:v0.10.0', 'fcn_resnet50', pretrained=True)
    

    model = seg(num_class=args.num_class, cfg=args, criterion=None, norm_layer=nn.BatchNorm2d)
    # for name in model.named_parameters:
    #     print('name:{}'.format(name))
    # 打印计算量并写入log
    flops, params = count_params(model)
    logger.info('Total FLOPs: {:.2f} GFLOPs'.format(flops / 1e9))
    logger.info('Total Params: {:.2f} M'.format(params / 1e6))
    # logger.info('Total params: {:.1f}M\n'.format(count_params(model)))
    logger.info('batch_size: {:}\n'.format(args.batch_size))
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # 自动分配数据到多个GPU上
        model = nn.DataParallel(model)

    # msrs_weight = torch.from_numpy(np.array(msrs_frq)).float().to(device)
    # cate_weight = torch.from_numpy(class_weight)
    # CEL_weighted = utils.CrossEntropyLoss2d(weight=msrs_weight)
    # msrs_edge = torch.from_numpy(np.array(edge_frq)).float()
    # edge_weighted = utils.BCELoss()
    # edge_weight = torch.from_numpy(bound_weight)
    # edge_weight_ = edge_weight.unsqueeze(1).unsqueeze(2)
    # edge_weight_ = edge_weight_.expand(args.batch_size, 2, 480, 640)
    # edge_weight1 = edge_weight.unsqueeze(1).unsqueeze(2)
    # edge_weight1 = edge_weight1.expand(1, 2, 480, 640)
    # weighted = utils.TotalLoss(weight=msrs_weight, weight_edge=edge_weight_)
    weighted = utils.TotalLoss()
    # weighted1 = utils.TotalLoss(weight=msrs_weight, weight_edge=edge_weight1)
    
    # CEL_weighted = utils.FocalLoss2d(weight=msrs_frq, gamma=2)
    model.train()
    model.to(device)
    weighted.to(device)
    # weighted1.to(device)
    # CEL_weighted.to('cpu')
    optimizer = torch.optim.AdamW([{"params": model.parameters()}, {"params": weighted.parameters()}], lr=args.lr, weight_decay=args.weight_decay)
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    global_step = 0
    # miou = 0
    previous_best = 0.0
    previous_acc_best = 0.0
    best_epoch = 0
    # remove_epoch = 0
    if args.latest_ckpt:
        global_step, args.start_epoch = load_ckpt(model, optimizer, args.latest_ckpt, device)

    # lr_decay_lambda = lambda epoch: 0.2 * args.lr_decay_rate ** ((epoch - args.start_epoch) // args.lr_epoch_per_decay)
    # 学习率衰减， 随着周期的增加而减小
    # (epoch - args.start_epoch) / (args.epochs - args.start_epoch)：从训练开始到现在经过的相对时间比例
    # lr_decay_lambda = lambda epoch: (1 - (epoch - args.start_epoch) / (args.epochs - args.start_epoch)) ** 0.8
    # 学习率调度器
    # scheduler = LambdaLR(optimizer, lr_lambda=lr_decay_lambda)
    scheduler = get_scheduler(optimizer, int(args.epochs + 1) * 196, 0.9, 196 * 10, 0.1)
    writer = SummaryWriter(args.ckpt_dir + args.summary_dir)
    local_count = 0
    for epoch in range(int(args.start_epoch), args.epochs):
        # if (epoch - args.start_epoch) % args.lr_epoch_per_decay == 0:
        lr = scheduler.get_lr()
        lr = sum(lr) / len(lr)
        # logger.info('===========> Epoch: {:}, LR: {:.7f}, Previous best: {:.3f}, Epoch best: {:}'.format(
        #     epoch, optimizer.param_groups[0]['lr'], previous_best, best_epoch))
        logger.info('===========> Epoch: {:}, LR: {:.7f}, Previous best: {:.3f}, Epoch best: {:}'.format(
            epoch, lr, previous_best, best_epoch))
        total_loss = 0.0
        # if epoch % args.save_epoch_freq == 0 and epoch != args.start_epoch:
        #     save_ckpt(args.ckpt_dir, model, optimizer, global_step, epoch,
        #               local_count, num_train)
        local_count += 1
        # if epoch > 9:
        #     flag=1
        #     msrs_weight = msrs_weight * ((1 - (epoch / args.epochs)) ** 0.6)
        #     if msrs_weight.max() < 1:
        #         print_count += 1
        #         msrs_weight = torch.ones_like(msrs_weight).float()
        #         if print_count > 1:
        #             flag = 0
        #     # msrs_weight.to(device)
        #     weighted = utils.TotalLoss(weight=msrs_weight)
        #     weighted.to(device)
        # msrs_weight = nn.Parameter(msrs_weight, requires_grad=True)
        for i, sample in enumerate(train_loader):
            model.train()
            # image = sample['image']
            image = sample['image'].to(device)
            thermal = sample['thermal'].to(device)
            target_scales = sample['label'].to(device)
            target_scales_edge = sample['label_edge'].to(device)
            
            # model = torch.hub.load('pytorch/vision:v0.10.0', 'fcn_resnet50', pretrained=True)
            # model.classifier[4] = nn.Conv2d(512, 9, kernel_size=1, stride=1)
            
            optimizer.zero_grad()
            pred_scales, pred_edge = model(image, thermal)
            # pred_scales, _ = model(image, thermal)
            # pred_scales = model(image)
            output = pred_scales.argmax(1)
            # output = output.cpu().numpy()
            metrics.update(output, target_scales.long())

            # 通道数为1时才有用
            pred_edge = pred_edge.squeeze(1)
            bc, h, w = target_scales.shape
            target = torch.zeros(bc, args.num_class, h, w)
            for c in range(args.num_class):
                target[:, c, :, :][target_scales == c] = 1
            target = target.to(device)

            # target_scales_edge /= 255

            target_edge = torch.zeros(bc, 2, h, w)
            # for c in range(2):
            #     target_edge[:, c, :, :][target_scales_edge == c] = 1
            target_edge[:, 0, :, :][target_scales_edge == 0] = 1
            target_edge[:, 1, :, :][target_scales_edge == 255] = 1
            target_edge = target_edge.to(device)
            
            # loss_all = CEL_weighted(pred_scales['out'], target)
            # loss_all = CEL_weighted(pred_scales, target)
            # loss_edge = edge_weighted(pred_scales_edge, target_scales_edge)

            # loss = loss_all + 0.1 * loss_edge
            # if bc == 1:
            #     loss = weighted1(pred_scales, target, pred_edge, target_edge)
            # else:
            loss = weighted(pred_scales, target, pred_edge, target_edge)
            # if loss_edge > loss_all:
            #     loss = loss_all * 0.3 + loss_edge * 0.7
            # else:
            #     loss = loss_all * 0.6 + loss_edge * 0.4
            # loss_edge = CEL_weighted(pred_scales_edge, target_edge)
            # with torch.autograd.detect_anomaly(True):
            loss.backward()
            # loss_edge.backward()
            
            # 检查是否有梯度消失问题
            # for name, param in model.named_parameters():
            #     if param.grad is not None:
            #         print(f'{name} gradient: {param.grad.mean()}')

            # 检查模型参数更新变化的大小
            # weights = model.state_dict()
            # for key in weights:
            #     print(weights[key])
            # 当前批次中处理的图像数量
            optimizer.step()
            scheduler.step()
            # weighted = utils.TotalLoss(msrs_weight)
            # if bc != 1:
            # if weighted.alpha.clamp(0.7, 1.2).item() not in all_alpha:
            #     all_alpha.append(weighted.alpha.clamp(0.7, 1.2).item())
            # if weighted.beta.clamp(0.0785, 0.3).item() not in all_beta:
            #     all_beta.append(weighted.beta.clamp(0.0785, 0.3).item())
            # elif bc == 1:
            #     if weighted1.alpha.item() not in all_alpha:
            #         all_alpha.append(weighted1.alpha.item())
            # local_count += image.data.shape[0]
            global_step += 1
            total_loss += loss.item()
            
            # if local_count > 10 and flag == 1:
            #     x0, x1, x2, x3, x4, x5, x6, x7, x8 = msrs_weight
            #     logger.info('catgory weight: background: {:}, car: {:}, person: {:}, bike: {:}, curve: {:}, car_stop: {:}, guardrail: {:}, color_cone: {:}, bump: {:}'.format(x0, x1, x2, x3, x4, x5, x6, x7, x8))
            #     flag = 0
            if i % 30 == 0:
                # if bc != 1:
                logger.info('Iters: {:}, Total loss: {:.3f}'.format(i, total_loss / (i+1)))
                # logger.info('Iters: {:}, Total loss: {:.3f}, alpha: {:.3f}, beta: {:.3f}'.format(i, total_loss / (i+1), weighted.alpha.clamp(0.7, 1.2).item(), weighted.beta.clamp(0.0785, 0.3)))
                # elif bc == 1:
                #     logger.info('Iters: {:}, Total loss: {:.3f}, alpha: {}, beta: {}'.format(i, total_loss / (i+1), weighted1.alpha.item(), 1 - (weighted1.alpha.item())))
            if global_step % args.print_freq == 0 or global_step == 1:
                # time_inter = time.time() - end_time
                writer.add_scalar('CrossEntropyLoss', loss.item(), global_step=global_step)
            
            if local_count % 5  == 0:
                args.visualize = True
            
            # 查看训练集的分割结果
            if args.visualize:
                output_ = output.cpu().numpy()
                visualize_result(output_, i, args.output)
                # args.visualize = False
               
                # count_inter = local_count - last_count
                # print_log(global_step, epoch, local_count, count_inter,
                #           num_train, loss, time_inter)
                # end_time = time.time()

                # last_count = local_count
        ious_train, miou_train = metrics.compute_iou()
        acc_train, macc_train = metrics.compute_pixel_acc()
        ious, MIOU, Accuracy, mAcc, loss_val = inference(args, model, g)
        if local_count % 5 == 0:
            bg_iou, car_iou, person_iou, bike_iou, curve_iou, car_stop_iou, guard_iou, color_cone_iou, bump_iou = ious
            bg_acc, car_acc, person_acc, bike_acc, curve_acc, car_stop_acc, guard_acc, color_cone_acc, bump_acc = Accuracy
            bg_iou_tr, car_iou_tr, person_iou_tr, bike_iou_tr, curve_iou_tr, car_stop_iou_tr, guard_iou_tr, color_cone_iou_tr, bump_iou_tr = ious_train
            bg_acc_tr, car_acc_tr, person_acc_tr, bike_acc_tr, curve_acc_tr, car_stop_acc_tr, guard_acc_tr, color_cone_acc_tr, bump_acc_tr = acc_train
            logger.info('***** Train *****')
            logger.info('IoU for each category >>> background:{:.2f}, car:{:.2f}, person:{:.2f}, bike:{:.2f}, curve:{:.2f}, car_stop:{:.2f}, guardrail:{:.2f}, color_cone:{:.2f}, bump:{:.2f}'.format(bg_iou_tr, car_iou_tr, person_iou_tr, bike_iou_tr, curve_iou_tr, car_stop_iou_tr, guard_iou_tr, color_cone_iou_tr, bump_iou_tr))
            logger.info('accuracy for each category >>> background:{:.2f}, car:{:.2f}, person:{:.2f}, bike:{:.2f}, curve:{:.2f}, car_stop:{:.2f}, guardrail:{:.2f}, color_cone:{:.2f}, bump:{:.2f}'.format(bg_acc_tr, car_acc_tr, person_acc_tr, bike_acc_tr, curve_acc_tr, car_stop_acc_tr, guard_acc_tr, color_cone_acc_tr, bump_acc_tr))
            logger.info("***** Evaluation *****")
            logger.info('IoU for each category >>> background:{:.2f}, car:{:.2f}, person:{:.2f}, bike:{:.2f}, curve:{:.2f}, car_stop:{:.2f}, guardrail:{:.2f}, color_cone:{:.2f}, bump:{:.2f}'.format(bg_iou, car_iou, person_iou, bike_iou, curve_iou, car_stop_iou, guard_iou, color_cone_iou, bump_iou))
            logger.info('accuracy for each category >>> background:{:.2f}, car:{:.2f}, person:{:.2f}, bike:{:.2f}, curve:{:.2f}, car_stop:{:.2f}, guardrail:{:.2f}, color_cone:{:.2f}, bump:{:.2f}'.format(bg_acc, car_acc, person_acc, bike_acc, curve_acc, car_stop_acc, guard_acc, color_cone_acc, bump_acc))
        logger.info('***** Train ***** >>>> meanIOU: {:.3f}, mAcc: {}\n'.format(miou_train, macc_train))
        logger.info('***** Evaluation ***** >>>> loss: {:.3f}, meanIOU: {:.3f}, mAcc: {}\n'.format(loss_val, MIOU, mAcc))

        if MIOU >= previous_best:
            if previous_best != 0:
                os.remove(os.path.join(args.ckpt_dir, 'epoch%d_%.3f.pth' % (best_epoch, previous_best)))
            previous_best = MIOU
            best_epoch = epoch
            save_ckpt(args.ckpt_dir, model, optimizer, global_step, best_epoch, MIOU)
        else:
            continue
        
        # if(miou<MIOU):
        #     # epoch_float = epoch + (local_count / num_train)
        #     # remove_epoch = epoch_float
        #     # ckpt_model_filename = "best{:0.3f}_ckpt_epoch_{:0.2f}.pth".format(miou,epoch_float)
        #     # path = os.path.join(args.ckpt_dir, ckpt_model_filename)
        #     # os.remove(path)
        #     miou = MIOU
        #     save_ckpt(args.ckpt_dir, model, optimizer, global_step, epoch,
        #               local_count, num_train,miou)
        

    # save_ckpt(args.ckpt_dir, model, optimizer, global_step, args.epochs,
    #           0, num_train)
    print("Training completed ")

if __name__ == '__main__':
    os.makedirs(args.ckpt_dir + args.summary_dir, exist_ok=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    # if not os.path.exists(args.ckpt_dir):
    #     os.mkdir(args.ckpt_dir)
    # if not os.path.exists(args.summary_dir):
    #     os.mkdir(args.summary_dir)
    train()
