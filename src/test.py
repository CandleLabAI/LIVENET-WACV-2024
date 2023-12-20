# ------------------------------------------------------------------------
# Copyright (c) 2023 CandleLabAI. All Rights Reserved.
# ------------------------------------------------------------------------

"""
Training code for LLIE with given configuration params

usage:

python test.py -opt cfs/lolv1.yaml
"""

# from models import LIENet, ResRefinement
# from torchvision.utils import save_image
# import numpy as np
# import torchvision
# import argparse
# # import config
# import time
# import time
# import math
# import cv2

import argparse
import logging
import os

from models import get_model, load_checkpoint
from PIL import Image
from glob import glob
from math import exp
import numpy as np
import lpips
import math

from torch.autograd import Variable
from torchvision import transforms
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torch

from utils import dict2str, parse, get_msg, select_device

curr_psnr = 0
curr_ssim = 0
curr_mae = 0
curr_lpips = 0

def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))

def _ssim(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class SSIM(nn.Module):
    def __init__(self, window_size = 11, size_average = True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            
            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def calculate_ssim(imgA, imgB):
	ssim = SSIM()
	score = ssim(imgA, imgB)
	return score

def calculate_mae(imgA, imgB):
	imgA = imgA.astype(np.float32)
	imgB = imgB.astype(np.float32)
	return np.sum(np.abs(imgA - imgB)) / np.sum(imgA + imgB)

def tensor(img):
    def to_4d(img):
        assert len(img.shape) == 3
        assert img.dtype == np.uint8
        img_new = np.expand_dims(img, axis=0)
        assert len(img_new.shape) == 4
        return img_new

    def to_CHW(img):
        return np.transpose(img, [2, 0, 1])

    def to_tensor(img):
        return torch.Tensor(img)

    return to_tensor(to_4d(to_CHW(img.astype(np.uint8)))) / 127.5 - 1

def calculate_lpips(imgA, imgB, model=None):
	model = lpips.LPIPS(net='alex')
	model.to("cpu")
	tA = tensor(imgA).to("cpu")
	tB = tensor(imgB).to("cpu")
	dist01 = model.forward(tA, tB).item()
	return dist01

def infer(low_img, normal_img, opt, generator, refiner):
    transform = transforms.Compose(
        [
            transforms.Resize((opt["datasets"]["val"]["img_size_h"], opt["datasets"]["val"]["img_size_w"])),
            transforms.ToTensor(),
        ])
    
    low_img = Image.open(low_img)
    low_img = transform(low_img)
    low_img = low_img.cuda().unsqueeze(0)

    normal_img = Image.open(normal_img)
    normal_img = transform(normal_img)
    normal_img = normal_img.cuda().unsqueeze(0)
    
    with torch.no_grad():
        pred_normal_image_gray, _, _, pred_refined_map = generator(low_img)
        pred_normal_image_rgb = refiner(pred_refined_map, pred_normal_image_gray)
    
    curr_psnr = calculate_psnr(normal_img[0].permute(1, 2, 0).detach().cpu().numpy() * 255.0, pred_normal_image_rgb[0].permute(1, 2, 0).detach().cpu().numpy() * 255.0)
    curr_ssim = calculate_ssim(normal_img, pred_normal_image_rgb)
    curr_mae = calculate_mae(normal_img[0].permute(1, 2, 0).detach().cpu().numpy() * 255.0, pred_normal_image_rgb[0].permute(1, 2, 0).detach().cpu().numpy() * 255.0)
    curr_lpips = calculate_lpips(normal_img[0].permute(1, 2, 0).detach().cpu().numpy() * 255.0, pred_normal_image_rgb[0].permute(1, 2, 0).detach().cpu().numpy() * 255.0)

    return curr_psnr, curr_ssim.detach().cpu().numpy().item(), curr_mae, curr_lpips

def parse_config():
    """
    Helper function to parse config
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-opt",
        type=str,
        required=True,
        help="Path to option YAML file."
    )
    args = parser.parse_args()

    opt = parse(args.opt)
    device = select_device()
    opt["device"] = device
    return opt

def init_log(opt):
    """
    Helper function for logging
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(opt["path"]["log_file"], mode="w"),
            logging.StreamHandler(),
        ],
    )
    logger = logging.getLogger(__name__)

    logger.info(get_msg())
    logger.info(dict2str(opt))

def main():
    """
    main function for LLIE training
    """
    opt = parse_config()
    init_log(opt)
    logger = logging.getLogger(__name__)

    with torch.no_grad():
        # get models
        generator, refiner = get_model(opt["model"], device=opt["device"])

        # opts
        if opt["train"]["optim"]["type"] == "Adam":
            opt_gen = optim.Adam(generator.parameters(), lr=float(opt["train"]["optim"]["lr"]))
            opt_ref = optim.Adam(refiner.parameters(), lr=float(opt["train"]["optim"]["lr"]))
        
        elif opt["train"]["optim"]["type"] == "SGD":
            opt_gen = optim.SGD(generator.parameters(), lr=float(opt["train"]["optim"]["lr"]))
            opt_ref = optim.SGD(refiner.parameters(), lr=float(opt["train"]["optim"]["lr"]))
        generator, refiner = load_checkpoint(
            generator, refiner, opt_gen, opt_ref, opt, logger
        )

        # get images
        test_low_images = sorted(glob(os.path.join(opt["datasets"]["val"]["low_images"], "*.png")))
        test_normal_images = sorted(glob(os.path.join(opt["datasets"]["val"]["normal_images"], "*.png")))
        
        # inference
        psnr = []
        ssim = []
        mae = []
        lpips = []
        for low_image, normal_image in zip(test_low_images, test_normal_images):
            curr_psnr, curr_ssim, curr_mae, curr_lpips = infer(low_image, normal_image, opt, generator, refiner)
            psnr.append(curr_psnr)
            ssim.append(curr_ssim)
            mae.append(curr_mae)
            lpips.append(curr_lpips)
        print("PSNR:", sum(psnr)/len(psnr))
        print("SSIM:", sum(ssim)/len(ssim))
        print("MAE:", sum(mae)/len(mae))
        print("LPIPS:", sum(lpips)/len(lpips))

if __name__ == "__main__":
    main()