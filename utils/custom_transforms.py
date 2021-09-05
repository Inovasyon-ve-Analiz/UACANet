import numpy as np
from PIL import Image
import cv2
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps, ImageFilter, ImageEnhance

from .utils import *

class resize:
    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        self.size = tuple(self.size)
        if 'image' in sample.keys():
            # print(sample['image'].dtype,"*"*20)

            sample['image'] = cv2.resize(sample['image'], self.size[:2])
            # print(sample['image'].dtype)
            # sample['image'] = sample['image'].resize(self.size, Image.BILINEAR)
        if 'gt' in sample.keys():
            sample['gt'] = cv2.resize(sample['gt'], self.size[:2])
            # sample['gt'] = sample['gt'].resize(self.size, Image.BILINEAR)
        
        return sample

class random_scale_crop:
    def __init__(self, range=[0.75, 1.25]):
        self.range = range

    def __call__(self, sample):
        scale_number = np.random.random()
        scale = scale_number * (self.range[1] - self.range[0]) + self.range[0]
        if np.random.random() < 0.5:
            for key in sample.keys():
                if key in ['image', 'gt']:
                    img = sample[key]
                    base_size = img.shape
                    scale_size = tuple((np.array(base_size) * scale).round().astype(int))
                    if scale_number >= 0.5:
                        img = cv2.resize(img, (scale_size[0],scale_size[1]))
                        if key == "image":
                            img = img[(img.shape[0] - base_size[1]) // 2:(img.shape[0] + base_size[0]) // 2, (img.shape[1] - base_size[1]) // 2:(img.shape[1] + base_size[1]) // 2,:]
                        else:
                            img = img[(img.shape[0] - base_size[1]) // 2:(img.shape[0] + base_size[0]) // 2, (img.shape[1] - base_size[1]) // 2:(img.shape[1] + base_size[1]) // 2]
                      
                    else:
                        if key == "image":
                            dtype = np.float32
                            img = cv2.resize(img, (scale_size[0],scale_size[1]))
                            zeros = np.zeros((base_size[0],base_size[1],3), dtype=dtype)
                            zeros[(base_size[0] - img.shape[0]) // 2:(img.shape[0] + base_size[0]) // 2, (base_size[1] - img.shape[1]) // 2:(img.shape[1] + base_size[1]) // 2,:] += img
                            img = zeros.copy()
                        else:
                            dtype = np.uint8
                            img = cv2.resize(img, (scale_size[0],scale_size[1]))
                            zeros = np.zeros((base_size[0],base_size[1]), dtype=dtype)
                            zeros[(base_size[0] - img.shape[0]) // 2:(img.shape[0] + base_size[0]) // 2, (base_size[1] - img.shape[1]) // 2:(img.shape[1] + base_size[1]) // 2] += img
                            img = zeros.copy()
                        
                    
                        # base_size = sample[key].size
                        # scale_size = tuple((np.array(base_size) * scale).round().astype(int))
                        # sample[key] = sample[key].resize(scale_size)
                        # sample[key] = sample[key].crop(((sample[key].size[0] - base_size[0]) // 2,
                        #                                 (sample[key].size[1] - base_size[1]) // 2,
                        #                                 (sample[key].size[0] + base_size[0]) // 2,
                        #                                 (sample[key].size[1] + base_size[1]) // 2))

        return sample

class random_flip:
    def __init__(self, lr=True, ud=True):
        self.lr = lr
        self.ud = ud

    def __call__(self, sample):
        lr = np.random.random() < 0.5 and self.lr is True
        ud = np.random.random() < 0.5 and self.ud is True

        for key in sample.keys():
            if key in ['image', 'gt']:

                if lr:
                    sample[key] = np.fliplr(sample[key])
                if ud:
                    sample[key] = np.flipud(sample[key])

        return sample

class random_rotate:
    def __init__(self, range=[0, 360], interval=1):
        self.range = range
        self.interval = interval

    def rotate_image(self, image, angle):
        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
        return result

    def __call__(self, sample):
        rot = (np.random.randint(*self.range) // self.interval) * self.interval
        rot = rot + 360 if rot < 0 else rot

        if np.random.random() < 0.5:
            for key in sample.keys():
                if key in ['image', 'gt']:
                    base_size = sample[key].shape[:2]

                    sample[key] = self.rotate_image(sample[key],rot)
                    sample[key] = sample[key][(sample[key].shape[0] - base_size[0]) // 2:(sample[key].shape[0] + base_size[0]) // 2,
                        (sample[key].shape[1] - base_size[1]) // 2:(sample[key].shape[1] + base_size[1]) // 2]

                    # sample[key] = sample[key].crop(((sample[key].size[0] - base_size[0]) // 2,
                    #                                 (sample[key].size[1] - base_size[1]) // 2,
                    #                                 (sample[key].size[0] + base_size[0]) // 2,
                    #                                 (sample[key].size[1] + base_size[1]) // 2))

        return sample

class random_image_enhance:
    def __init__(self, methods=['contrast', 'brightness', 'sharpness']):
        self.enhance_method = []
        if 'contrast' in methods:
            self.enhance_method.append(ImageEnhance.Contrast)
        if 'brightness' in methods:
            self.enhance_method.append(ImageEnhance.Brightness)
        if 'sharpness' in methods:
            self.enhance_method.append(ImageEnhance.Sharpness)

    def __call__(self, sample):
        # image = sample['image']
        # np.random.shuffle(self.enhance_method)

        # for method in self.enhance_method:
        #     if np.random.random() > 0.5:
        #         enhancer = method(image)
        #         factor = float(1 + np.random.random() / 10)
        #         image = enhancer.enhance(factor)
        # sample['image'] = image

        return sample

class random_dilation_erosion:
    def __init__(self, kernel_range):
        self.kernel_range = kernel_range

    def __call__(self, sample):
        gt = sample['gt']
        # gt = np.array(gt)
        key = np.random.random()
        # kernel = np.ones(tuple([np.random.randint(*self.kernel_range)]) * 2, dtype=np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (np.random.randint(*self.kernel_range), ) * 2)
        if key < 1/3:
            gt = cv2.dilate(gt, kernel)
        elif 1/3 <= key < 2/3:
            gt = cv2.erode(gt, kernel)

        sample['gt'] = gt

        return sample

class random_gaussian_blur:
    def __init__(self):
        pass

    def __call__(self, sample):
        # image = sample['image']
        # if np.random.random() < 0.5:
        #     image = cv2.GaussianBlur(image, (11,11))
        #     image.filter(ImageFilter.GaussianBlur(radius=np.random.random()))
        # sample['image'] = image

        return sample

class tonumpy:
    def __init__(self):
        pass

    def __call__(self, sample):
        image = sample['image']
        gt = sample['gt']
        sample['image'] = np.array(image, dtype=np.float32)
        sample['gt'] = np.array(gt, dtype=np.float32)
        # sample['image'].dtype=np.float32
        # sample['gt'].dtype=np.float32
        
        return sample

class normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        image, gt = sample['image'], sample['gt']
        image /= 2**16
        image -= self.mean
        image /= self.std

        gt /= 255
        sample['image'] = image
        sample['gt'] = gt

        return sample

class totensor:
    def __init__(self):
        pass

    def __call__(self, sample):
        image, gt = sample['image'], sample['gt']
        image = image.transpose((2, 0, 1))
        image = torch.from_numpy(image).float()
        
        gt = torch.from_numpy(gt)
        gt = gt.unsqueeze(dim=0)

        sample['image'] = image
        sample['gt'] = gt
        # print(image.shape)
        # print(gt.shape)
        return sample
