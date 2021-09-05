import os
import cv2

import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms

from PIL import Image
import SimpleITK as sitk
from utils.custom_transforms import *

from windowing import output

class PolypDataset(data.Dataset):

    def rotate_image(self, image, angle):
        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
        return result

    def augmentation(self, image, type):

        if type == "rotated":
            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        elif type == "rotated45":
            image = self.rotate_image(image, 45)
        elif type == "hflip":
            image = cv2.flip(image,0)
        elif type == "vflip":
            image = cv2.flip(image,1)

        return image

    """
    dataloader for polyp segmentation tasks
    """
    def __init__(self, image_root, gt_root, opt):
        self.images = [os.path.join(image_root, f) for f in os.listdir(image_root) if f.endswith('.dcm')]
        self.gts = [os.path.join(gt_root, f) for f in os.listdir(gt_root) if f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.filter_files()
        self.size = len(self.images)

        tfs = []
        for key, value in zip(opt.transforms.keys(), opt.transforms.values()):
            if value is not None:
                tf = eval(key)(**value)
            else:
                tf = eval(key)()
            tfs.append(tf)
        self.transform = transforms.Compose(tfs)

    def __getitem__(self, index):
        image = self.dcm_loader(self.images[index][0], self.images[index][1])
        gt = self.binary_loader(self.gts[index][0], self.gts[index][1])
        
        original_size = gt.shape
        name = self.images[index][0].split('/')[-1] + str(self.images[index][1])
        
        sample = self.transform({'image': image, 'gt': gt})
        sample['name'] = name
        sample['original_size'] = original_size
        return sample

    def filter_files(self):
        assert len(self.images) == len(self.gts)
        images = []
        gts = []
        for img_path, gt_path in zip(self.images, self.gts):
            img = self.dcm_loader(img_path, 0)
            gt = self.binary_loader(gt_path, 0)
            
            if img.shape[:2] == gt.shape[:2]:
                images.append((img_path, 0))
                images.append((img_path, "a"))
                images.append((img_path, "v"))
                images.append((img_path, "r"))
                images.append((img_path, "ah"))
                images.append((img_path, "rh"))
                
                gts.append((gt_path, 0))
                gts.append((gt_path, "a"))
                gts.append((gt_path, "v"))
                gts.append((gt_path, "r"))
                gts.append((gt_path, "ah"))
                gts.append((gt_path, "rh"))

        self.images = images
        self.gts = gts

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')
    
    def dcm_loader(self, path, atype=0):
        pixels = output(path)

        if atype == "a":
            pixels = self.augmentation(pixels, "rotated")
        elif atype == "v":
            pixels = self.augmentation(pixels, "vflip")
        elif atype == "r":
            pixels = self.augmentation(pixels, "rotated45")
        elif atype == "ah":
            pixels = self.augmentation(pixels, "rotated")
            pixels = self.augmentation(pixels, "hflip")
        elif atype == "rh":
            pixels = self.augmentation(pixels, "rotated45")
            pixels = self.augmentation(pixels, "hflip")
        
        return pixels

    def binary_loader(self, path, atype=0):
        img = cv2.imread(path,0)

        if atype == "a":
            img = self.augmentation(img, "rotated")
        elif atype == "v":
            img = self.augmentation(img, "vflip")
        elif atype == "r":
            img = self.augmentation(img, "rotated45")
        elif atype == "ah":
            img = self.augmentation(img, "rotated")
            img = self.augmentation(img, "hflip")
        elif atype == "rh":
            img = self.augmentation(img, "rotated45")
            img = self.augmentation(img, "hflip")

        return img

    def __len__(self):
        return self.size
