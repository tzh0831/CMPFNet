import numpy as np
from torch.utils.data import Dataset
import matplotlib
import matplotlib.colors
import skimage.transform
import random
import torchvision
import torch
from PIL import Image
import os
import matplotlib.pyplot as plt
import cv2
import torchvision.transforms.functional as TF
import sys
sys.path.append('/workspace/try1')
# 用于重新加载一个已经加载的模块
# import importlib
# importlib.reload(sys)

image_h = 720
image_w = 1280

train_rgb_file = '/workspace/try1/MFNet/train/rgb'
train_the_file = '/workspace/try1/MFNet/train/thermal'
train_label_file = '/workspace/try1/MFNet/train/label'
train_label_edge_file = '/workspace/try1/MFNet/train/label_edge'
test_rgb_file = '/workspace/try1/PST900/test/rgb'
test_the_file = '/workspace/try1/PST900/test/thermal'
test_label_file = '/workspace/try1/PST900/test/labels'
test_label_edge_file = '/workspace/try1/show/label_edge'

# train_rgb_file = '/workspace/try1/PST900/train/rgb'
# train_the_file = '/workspace/try1/PST900/train/thermal'
# train_label_file = '/workspace/try1/PST900/train/labels'
# test_rgb_file = '/workspace/try1/PST900/test/rgb'
# test_the_file = '/workspace/try1/PST900/test/thermal'
# test_label_file = '/workspace/try1/PST900/test/labels'

class MSRS(Dataset):
    def __init__(self, transform=None, phase_train=True, data_dir=None):
        self.phase_train = phase_train
        self.transform = transform
        self.rgb_train = train_rgb_file
        self.the_train = train_the_file
        self.label_train = train_label_file
        self.label_edge_train = train_label_edge_file
        self.rgb_test = test_rgb_file
        self.the_test = test_the_file
        self.label_test = test_label_file
        self.label_edge_test = test_label_edge_file
        
    def __len__(self):
        if self.phase_train:
            return len(os.listdir(self.rgb_train))
        else:
            return len(os.listdir(self.rgb_test))

    def __getitem__(self, idx):
        if self.phase_train:
            rgb_dir = self.rgb_train
            the_dir = self.the_train
            label_dir = self.label_train
            label_edge_dir = self.label_edge_train
            path = '/workspace/try1/PST900/train/'
        else:
            rgb_dir = self.rgb_test
            the_dir = self.the_test
            label_dir = self.label_test
            # label_edge_dir = self.label_edge_test
            path = '/workspace/try1/PST900/test/'
        # label_edge_path = os.listdir(label_edge_dir)
        # label_edge_path = sorted(label_edge_path, key=lambda x:(len(x), x), reverse=False)
        label_path = os.listdir(label_dir)
        label_path = sorted(label_path, key=lambda x: (len(x), x), reverse=False)
        the_path = os.listdir(the_dir)
        the_path = sorted(the_path, key=lambda x: (len(x), x), reverse=False)
        rgb_path = os.listdir(rgb_dir)
        rgb_path = sorted(rgb_path, key=lambda x: (len(x), x), reverse=False)
        # print(label_path[idx])
        # label_edge = np.array(Image.open(path + 'label_edge/' + label_edge_path[idx]))
        label = np.array(Image.open(path + 'labels/' + label_path[idx]))
        thermal = np.array(Image.open(path + 'thermal/' + the_path[idx])) / 255
        rgb = np.array(Image.open(path + 'rgb/' + rgb_path[idx])) / 255
        output_array = np.zeros((thermal.shape[0], thermal.shape[1], 3))
        # 单通道图像数据复制到多通道数组
        output_array[ :, :, 0] = thermal[ :, :]  # 复制第一个通道的数据到新数组的第一个通道
        output_array[ :, :, 1] = thermal[ :, :]  # 复制第一个通道的数据到新数组的第二个通道
        output_array[ :, :, 2] = thermal[ :, :]  # 复制第一个通道的数据到新数组的第三个通道
        # sample = {'image': rgb, 'thermal': output_array, 'label': label, 'label_edge': label_edge}
        sample = {'image': rgb, 'thermal': output_array, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        return sample

# 增加数据集多样性，提高模型对不同颜色条件的适应能力
class RandomHSV(object):
    """
        Args:
            h_range (float tuple): random ratio of the hue channel,
                new_h range from h_range[0]*old_h to h_range[1]*old_h.
            s_range (float tuple): random ratio of the saturation channel,
                new_s range from s_range[0]*old_s to s_range[1]*old_s.
            v_range (int tuple): random bias of the value channel,(亮度或明暗程度)
                new_v range from old_v-v_range to old_v+v_range.
        Notice:
            h range: 0-1
            s range: 0-1
            v range: 0-255
    """
    # def __init__(self, h_range, s_range, v_range):
    #     assert isinstance(h_range, (list, tuple)) and \
    #            isinstance(s_range, (list, tuple)) and \
    #            isinstance(v_range, (list, tuple))
    #     self.h_range = h_range
    #     self.s_range = s_range
    #     self.v_range = v_range
    #     # self.p = p
    # def __init__(self):
        # self.p = p

    def __call__(self, sample):
        # r = random.random()
        # img = sample['image']
        # img_ = torchvision.transforms.functional.to_pil_image(img)
        # img_.save('/workspace/try1/MFNet/mid/hsv_pre.png')
        # if r < self.p:
        self.brightness = random.uniform(0.5, 1.5)
        sample['image'] = TF.adjust_brightness(sample['image'], self.brightness)
        self.contrast = random.uniform(0.5, 1.5)
        sample['image'] = TF.adjust_contrast(sample['image'], self.contrast)
        self.saturation = random.uniform(0.5, 1.5)
        sample['image'] = TF.adjust_saturation(sample['image'], self.saturation)
        # image = sample['image']
        # image_ = torchvision.transforms.functional.to_pil_image(image)
        # image_.save('/workspace/try1/MFNet/mid/hsv.png')
        # img = sample['image']
        # cv2.imwrite(os.path.join('/workspace/try1/MFNet/mid', 'hsv_pre.png'), img * 255)
        # img_hsv = matplotlib.colors.rgb_to_hsv(img)
        # img_h, img_s, img_v = img_hsv[:, :, 0], img_hsv[:, :, 1], img_hsv[:, :, 2]
        # # 生成在min和max之间的一个随机浮点数（包括min但不包括max）
        # h_random = np.random.uniform(min(self.h_range), max(self.h_range))
        # s_random = np.random.uniform(min(self.s_range), max(self.s_range))
        # v_random = np.random.uniform(min(self.v_range), max(self.v_range))
        # # np.clip()将结果限定在0-1（255）范围内
        # img_h = np.clip(img_h + h_random, 0, 1)
        # img_s = np.clip(img_s * s_random, 0, 1)
        # img_v = np.clip(img_v * v_random, 0, 1)
        # # img_h = np.clip(img_h * h_random, 0.5, 1.5)
        # # img_s = np.clip(img_s * s_random, 0.5, 100)
        # # img_v = np.clip(img_v + v_random, 0, 255)
        # # 重新组合成三维数组
        # img_hsv = np.stack([img_h, img_s, img_v], axis=2)
        # img_new = matplotlib.colors.hsv_to_rgb(img_hsv)
        # # img = np.array(image_data, dtype=np.float64)
        # cv2.imwrite(os.path.join('/workspace/try1/MFNet/mid', 'hsv.png'), img_new * 255)
        # # plt.imshow(img_new)
        # # plt.show()
        # return {'image': img_new, 'thermal': sample['thermal'], 'label': sample['label'], 'label_edge': sample['label_edge']}
        # if random.random() < self.p:
        #     self.brightness = random.uniform(0.5, 1.5)
        #     sample['image'] = TF.adjust_brightness(sample['image'], self.brightness)
        #     self.contrast = random.uniform(0.5, 1.5)
        #     sample['image'] = TF.adjust_contrast(sample['image'], self.contrast)
        #     self.saturation = random.random(0.5, 1.5)
        #     sample['image'] = TF.adjust_saturation(sample['image'], self.saturation)
        return sample

# class scaleNorm(object):
#     def __call__(self, sample):
#         image, thermal, label, label_edge = sample['image'], sample['thermal'], sample['label'], sample['label_edge']
#         # image, thermal, label = sample['image'], sample['thermal'], sample['label']
#         # (image_h, image_w):目标图像的高和宽
#         # order=1:采用双线性插值
#         # mode=reflect:反射边界(镜像)条件来填充
#         image = skimage.transform.resize(image, (image_h, image_w), order=1, mode='reflect', preserve_range=True)
#         # 最近邻插值
#         thermal = skimage.transform.resize(thermal, (image_h, image_w), order=0, mode='reflect', preserve_range=True)
#         label = skimage.transform.resize(label, (image_h, image_w), order=0, mode='reflect', preserve_range=True)
#         label_edge = skimage.transform.resize(label_edge, (image_h, image_w), order=0, mode='reflect', preserve_range=True)
#         return {'image': image, 'thermal': thermal, 'label': label, 'label_edge': label_edge}

# class RandomScale(object):
#     def __init__(self, scale):
#         self.scale_low = min(scale)#1.0
#         self.scale_high = max(scale)#1.4

#     def __call__(self, sample):
#         image, thermal, label, label_edge = sample['image'], sample['thermal'], sample['label'], sample['label_edge']
#         # image, thermal, label = sample['image'], sample['thermal'], sample['label']
#         # 1.0-1.4之间随机生成一个数
#         target_scale = random.uniform(self.scale_low, self.scale_high)
#         # (H, W, C)
#         # 先四舍五入然后int成整数
#         target_height = int(round(target_scale * image.shape[0]))
#         target_width = int(round(target_scale * image.shape[1]))
#         image = skimage.transform.resize(image, (target_height, target_width), order=1, mode='reflect', preserve_range=True)
#         thermal = skimage.transform.resize(thermal, (target_height, target_width), order=0, mode='reflect', preserve_range=True)
#         label = skimage.transform.resize(label, (target_height, target_width), order=0, mode='reflect', preserve_range=True)
#         label_edge = skimage.transform.resize(label_edge, (target_height, target_width), order=0, mode='reflect', preserve_range=True)
#         return {'image': image, 'thermal': thermal, 'label': label, 'label_edge': label_edge}

# 随机裁剪
# class RandomCrop(object):
#     def __init__(self, crop_rate=0.1, prob=1.0):
#         # self.size = size
#         # self.scale = scale
#         self.crop_rate = crop_rate
#         self.prob = prob

#     def __call__(self, sample):
#         # H, W = sample['image'].shape[:2]
#         # tH, tW = self.size
#         # ratio = random.random() * (self.scale[1] - self.scale[0]) + self.scale[0]
#         # scale = int(tH * ratio), int(tW * 4 * ratio)
#         # scale_factor = min(max(scale) / max(H, W), min(scale) / min(H, W))
#         # nH, nW = int(H * scale_factor + 0.5), int(W * scale_factor + 0.5)
#         # for k, v in sample.items():
#         #     if k == 'label' or k == 'label_edge':
#         #         sample[k] = TF.resize(v, (nH, nW), TF.InterpolationMode.NEAREST)
#         #     else:
#         #         sample[k] = TF.resize(v, (nH, nW), TF.InterpolationMode.BILINEAR)
#         # margin_h = max(sample['image'].shape[1] - tH, 0)
#         # margin_w = max(sample['image'].shape[2] - tW, 0)
#         # y1 = random.randint(0, margin_h + 1)
#         # x1 = random.randint(0, margin_w + 1)
#         # y2 = y1 + tH
#         # x2 = x1 + tW
#         # for k, v in sample.items():
#         #     sample[k] = v[:, y1:y2, x1:x2]
#         # if sample['image'].shape[1:] != self.size:
#         #     padding = [0, 0, tW - sample['image'].shape[2], tH - sample['image'].shape[1]]
#         #     for k, v in sample.items():
#         #         if k == 'label' or k == 'label_edge':
#         #             sample[k] = TF.pad(v, padding)
#         #         else:
#         #             sample[k] = TF.pad(v, padding)
#         # return sample
#         image, thermal, label, label_edge = sample['image'], sample['thermal'], sample['label'], sample['label_edge']
#         # cv2.imwrite(os.path.join('/workspace/try1/MFNet/mid', 'crop_pre.png'), image * 255)
#         # cv2.imwrite(os.path.join('/workspace/try1/MFNet/mid', 'crop_pre_the.png'), thermal * 255)
#         w, h, c = image.shape
#             # h1 = np.random.randint(0, h*self.crop_rate)
#             # w1 = np.random.randint(0, w*self.crop_rate)
#         h1 = np.random.randint(0, h // 2)
#         w1 = np.random.randint(0, w // 2)
#             # h2 = np.random.randint(h-h*self.crop_rate, h+1)
#             # w2 = np.random.randint(w-w*self.crop_rate, w+1)
#             # h2 = int(h1 + h*self.crop_rate)
#             # w2 = int(w1 + w*self.crop_rate)
#         h2 = np.random.randint(h1 + 50, h1 + h // 2)
#         w2 = np.random.randint(w1 + 50, w1 + w // 2)
#             # sub_image = image[w1:w2, h1:h2, :]
#             # sub_image = cv2.resize(sub_image, (h2-h1, w2-w1), cv2.INTER_LINEAR)
#         image[w1:w2, h1:h2, :] = 0
#         thermal[w1:w2, h1:h2, :] = 0
#         label[w1:w2, h1:h2] = 0
#         label_edge[w1:w2, h1:h2] = 0
#         # cv2.imwrite(os.path.join('/workspace/try1/MFNet/mid', 'crop_rgb.png'), image * 255)
#         # cv2.imwrite(os.path.join('/workspace/try1/MFNet/mid', 'crop_the.png'), thermal * 255)
#         return {'image': image, 'thermal': thermal, 'label': label, 'label_edge': label_edge}
#         # image, thermal, label = sample['image'], sample['thermal'], sample['label']
#         # return {'image': image[i: i + image_h, j: j + image_w, :], 'thermal': thermal[i: i + image_h, j: j + image_w, :], 'label': label[i: i + image_h, j: j + image_w], 'label_edge': label_edge[i: i + image_h, j: j + image_w]}

# 随机水平翻转
class RandomFlip(object):
    # def __init__(self, p=0.5):
    #     self.p = p
        
    def __call__(self, sample):
        # if random.random() < self.p:
        #     for k ,v in sample.items():
        #         sample[k] = TF.hflip(v)
        #         return sample
        # return sample
        image, thermal, label, label_edge = sample['image'], sample['thermal'], sample['label'], sample['label_edge']
        # image, thermal, label = sample['image'], sample['thermal'], sample['label']
        # 生成一个[0, 1)的随机数，即有50%的概率if条件为True
        if random.random() < 0.5:
            # 每一行像素水平翻转（被镜像），翻转操作生成一个新的数组
            image = np.fliplr(image).copy()
            thermal = np.fliplr(thermal).copy()
            label = np.fliplr(label).copy()
            label_edge = np.fliplr(label_edge).copy()
        # return {'image': image, 'thermal': thermal, 'label': label}
        return {'image': image, 'thermal': thermal, 'label': label, 'label_edge': label_edge}

# Transforms on torch.*Tensor
class Normalize(object):
    def __call__(self, sample):
        image, thermal = sample['image'], sample['thermal']
        # image_ = torchvision.transforms.functional.to_pil_image(image)
        # image_.save('/workspace/try1/MFNet/mid/norm_pre.png')
        # thermal_ = torchvision.transforms.functional.to_pil_image(thermal)
        # thermal_.save('/workspace/try1/MFNet/mid/norm_pre_the.png')
        # image = torchvision.transforms.Normalize(mean=[0.221, 0.259, 0.230], std=[0.228, 0.231, 0.236])(image)
        # image = torchvision.transforms.Normalize(mean=[0.222, 0.259, 0.213], std=[0.249, 0.284, 0.289])(image)
        # image = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image)
        image = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image)
        thermal = torchvision.transforms.Normalize(mean=[0.430, 0.430, 0.430], std=[0.226, 0.226, 0.226])(thermal)
        # thermal = torchvision.transforms.Normalize(mean=[0.395, 0.395, 0.395], std=[0.184, 0.184, 0.184])(thermal)
        # thermal = torchvision.transforms.Normalize(mean=[0.449, 0.449, 0.449], std=[0.226, 0.226, 0.226])(thermal)
        sample['image'] = image
        sample['thermal'] = thermal
        # image = torchvision.transforms.functional.to_pil_image(image)
        # image.save('/workspace/try1/MFNet/mid/norm_rgb.png')
        # thermal = torchvision.transforms.functional.to_pil_image(thermal)
        # thermal.save('/workspace/try1/MFNet/mid/norm_the.png')
        return sample

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        image, thermal, label, label_edge = sample['image'], sample['thermal'], sample['label'], sample['label_edge']
        # image, thermal, label = sample['image'], sample['thermal'], sample['label']
        # Generate different label scales
        # 不同分辨率的标签图像可以学习到全局和局部的特征，有利于提高模型泛化能力
        # label2 = skimage.transform.resize(label, (label.shape[0] // 2, label.shape[1] // 2), order=0, mode='reflect', preserve_range=True)
        # label3 = skimage.transform.resize(label, (label.shape[0] // 4, label.shape[1] // 4), order=0, mode='reflect', preserve_range=True)
        # label4 = skimage.transform.resize(label, (label.shape[0] // 8, label.shape[1] // 8), order=0, mode='reflect', preserve_range=True)
        # label5 = skimage.transform.resize(label, (label.shape[0] // 16, label.shape[1] // 16), order=0, mode='reflect', preserve_range=True)
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        thermal = thermal.transpose((2, 0, 1)).astype(np.float64)
        # return {'image': torch.from_numpy(image).float(),
        #         'thermal': torch.from_numpy(thermal).float(),
        #         'label': torch.from_numpy(label).float()}
        return {'image': torch.from_numpy(image).float(),
                'thermal': torch.from_numpy(thermal).float(),
                'label': torch.from_numpy(label).float(),
                'label_edge': torch.from_numpy(label_edge).float()}

# class RandomCrop():
#     def __init__(self, crop_rate=0.1, prob=1.0):
#         super(RandomCrop, self).__init__()
#         self.crop_rate = crop_rate
#         self.prob      = prob

#     def __call__(self, sample):
#         image, thermal, label, label_edge = sample['image'], sample['thermal'], sample['label'], sample['label_edge']
#         cv2.imwrite(os.path.join('/workspace/try1/MFNet/mid', 'crop_pre.png'), image * 255)
#         cv2.imwrite(os.path.join('/workspace/try1/MFNet/mid', 'crop_pre_the.png'), thermal * 255)
#         if np.random.rand() < self.prob:
#             w, h, c = image.shape

#             h1 = np.random.randint(0, h*self.crop_rate)
#             w1 = np.random.randint(0, w*self.crop_rate)
#             h2 = np.random.randint(h-h*self.crop_rate, h+1)
#             w2 = np.random.randint(w-w*self.crop_rate, w+1)

#             image = image[w1:w2, h1:h2]
#             thermal = thermal[w1:w2, h1:h2]
#             label = label[w1:w2, h1:h2]
#             label_edge = label_edge[w1:w2, h1:h2]
#         cv2.imwrite(os.path.join('/workspace/try1/MFNet/mid', 'crop_rgb.png'), image * 255)
#         cv2.imwrite(os.path.join('/workspace/try1/MFNet/mid', 'crop_the.png'), thermal * 255)
#         return {'image': image, 'thermal': thermal, 'label': label, 'label_edge': label_edge}

class RandomResizedCrop:
    def __init__(self, size, scale) -> None:
        """Resize the input image to the given size.
        """
        self.size = size
        self.scale = scale
        # self.seg_fill = seg_fill

    def __call__(self, sample: list) -> list:
        # img, mask = sample['img'], sample['mask']
        # image, thermal = sample['image'], sample['thermal']
        # image = torchvision.transforms.functional.to_pil_image(image)
        # image.save('/workspace/try1/MFNet/mid/crop_pre.png')
        # thermal = torchvision.transforms.functional.to_pil_image(thermal)
        # thermal.save('/workspace/try1/MFNet/mid/crop_pre_the.png')
        H, W = sample['image'].shape[1:]
        tH, tW = self.size
        # get the scale
        ratio = random.random() * (self.scale[1] - self.scale[0]) + self.scale[0]
        # ratio = random.uniform(min(self.scale), max(self.scale))
        scale = int(tH*ratio), int(tW*4*ratio)
        # scale the image 
        scale_factor = min(max(scale)/max(H, W), min(scale)/min(H, W))
        nH, nW = int(H * scale_factor + 0.5), int(W * scale_factor + 0.5)
        # nH, nW = int(math.ceil(nH / 32)) * 32, int(math.ceil(nW / 32)) * 32
        # scale = random.uniform(self.scale[0], self.scale[1])
        # nH, nW = (int(round(H * scale)), int(round(W * scale)))
        for k, v in sample.items():
        #     if k == 'image': 
        #         sample[k] = TF.resize(v, (nH, nW), TF.InterpolationMode.BILINEAR)
        #     else:
        #         if k == 'label' or k == 'label_edge':
        #             v = v.reshape(1, H, W)               
        #         sample[k] = TF.resize(v, (nH, nW), TF.InterpolationMode.NEAREST)
            if k == 'label' or k == 'label_edge': 
            # if k == 'label': 
              v = v.reshape(1, H, W)               
              sample[k] = TF.resize(v, (nH, nW), TF.InterpolationMode.NEAREST)
            else:
                sample[k] = TF.resize(v, (nH, nW), TF.InterpolationMode.BILINEAR)

        # random crop
        margin_h = max(sample['image'].shape[1] - tH, 0)
        margin_w = max(sample['image'].shape[2] - tW, 0)
        y1 = random.randint(0, margin_h+1)
        x1 = random.randint(0, margin_w+1)
        y2 = y1 + tH
        x2 = x1 + tW
        for k, v in sample.items():
            sample[k] = v[:, y1:y2, x1:x2]
        # pad the image
        if sample['image'].shape[1:] != self.size:
            padding = [0, 0, tW - sample['image'].shape[2], tH - sample['image'].shape[1]]
            for k, v in sample.items():
                if k == 'label' or k == 'label_edge':
                # if k == 'label':
                    # v = v.reshape(1, nH, nW)             
                    sample[k] = TF.pad(v, padding, fill=0)
                    sample[k] = sample[k].reshape(H, W)
                else:
                    sample[k] = TF.pad(v, padding, fill=0)
        else:
            for k, v in sample.items():
                # if k == 'label':
                if k == 'label' or k == 'label_edge':
                    sample[k] = sample[k].reshape(H, W)
        # image = sample['image']
        # thermal = sample['thermal']
        # image = torchvision.transforms.functional.to_pil_image(image)
        # image.save('/workspace/try1/MFNet/mid/crop_rgb.png')
        # thermal = torchvision.transforms.functional.to_pil_image(thermal)
        # thermal.save('/workspace/try1/MFNet/mid/crop_the.png')
        return sample
