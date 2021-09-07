import torch
from torch.nn import modules
import yaml
import os
import argparse
import tqdm
import sys

import torch.nn.functional as F
import numpy as np

from PIL import Image
from easydict import EasyDict as ed

filepath = os.path.split(__file__)[0]
repopath = os.path.split(filepath)[0]
sys.path.append(repopath)

from lib import *
from utils.dataloader import *
from utils.utils import *

def _args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/UACANet-L.yaml')
    parser.add_argument('--dataset', type=str, default="new_dataset")
    parser.add_argument('--inme', type=str, default="ISKEMI")
    parser.add_argument('--epoch', type=int, default=100) pth_path
    parser.add_argument('--pth_path', type=str, default="/kaggle/working/models")
    
    return parser.parse_args()

def test(opt, args):
    model = eval(opt.Model.name)(opt.Model)
    pth_path = os.path.join(opt.Test.pth_path, args.dataset, args.inme, "UACANet_" + args.inme + "_" + str(args.epoch) + "_Epoch.pth")
    model.load_state_dict(torch.load(pth_path))
    model.cuda()
    model.eval()    

    print('#' * 20, 'Test prep done, start testing', '#' * 20)

    data_path = os.path.join(opt.Test.gt_path, args.inme, "test")
    save_path = os.path.join(opt.Test.out_path, args.inme, "test", "RESULTS")

    os.makedirs(save_path, exist_ok=True)
    image_root = os.path.join(data_path, 'PNG')
    gt_root = os.path.join(data_path, 'MASKS')
    test_dataset = PolypDataset(image_root, gt_root, opt.Test)
    test_loader = data.DataLoader(dataset=test_dataset,
                                  batch_size=1,
                                  num_workers=opt.Test.num_workers,
                                  pin_memory=opt.Test.pin_memory)

    for i, sample in tqdm.tqdm(enumerate(test_loader), total=len(test_loader), position=1, leave=False, bar_format='{desc:<30}{percentage:3.0f}%|{bar:50}{r_bar}'):
        image = sample['image']
        name = sample['name']
        original_size = sample['original_size']

        image = image.cuda()
        out = model(image)['pred']
        
        out = F.interpolate(out, original_size, mode='bilinear', align_corners=True)
        out = out.data.sigmoid().cpu().numpy().squeeze()
        out = (out - out.min()) / (out.max() - out.min() + 1e-8)
        Image.fromarray(((out > 0.5) * 255).astype(np.uint8)).save(os.path.join(save_path, name[0].split(".")[0]+".png"))

    print('#' * 20, 'Test done', '#' * 20)

if __name__ == "__main__":
    args = _args()
    opt = ed(yaml.load(open(args.config), yaml.FullLoader))
    if "dicom" in args.dataset:
        opt.Test.transforms.resize.size = [352, 352, 3]
    test(opt, args)
