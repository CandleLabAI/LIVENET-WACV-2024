# ------------------------------------------------------------------------
# Copyright (c) 2023 CandleLabAI. All Rights Reserved.
# ------------------------------------------------------------------------

"""
Training code for LLIE with given configuration params

usage:

python train.py -opt cfs/lolv1.yaml
"""

import argparse
import logging
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from tqdm import tqdm
import torchvision
import torch

from models import get_model, initialize_weights, GenLoss, RefineLoss, load_checkpoint, save_checkpoint, get_refined_image, calculate_psnr
from utils import dict2str, parse, get_msg, seed_everything, select_device
from data import get_data

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

train_step = 0
val_step = 0

def train_fn(generator, refiner, train_dataloader, opt_gen, opt_ref, gen_loss, ref_loss, train_writer, opt):
    loop = tqdm(train_dataloader)
    curr_psnr = 0
    global train_step
    
    generator.train()
    refiner.train()
    
    for batch_idx, (low_image_rgb, normal_image_rgb, normal_image_gray) in enumerate(loop):
        low_image_rgb = low_image_rgb.to(device=opt["device"])
        normal_image_rgb = normal_image_rgb.to(device=opt["device"])
        normal_image_gray = normal_image_gray.to(device=opt["device"])

        # get ground truth of trasmission map and atmospheric light
        orig_transmission_map, orig_atmosphere_light, orig_refined_image = get_refined_image(low_image_rgb, normal_image_rgb, opt)

        pred_normal_image_gray, pred_transmission_map, pred_atmosphere_light, pred_refined_map = generator(low_image_rgb)
        pred_normal_image_rgb = refiner(pred_refined_map.detach(), pred_normal_image_gray.detach())
        
        total_loss = gen_loss(normal_image_gray, orig_transmission_map, orig_atmosphere_light, pred_normal_image_gray, pred_transmission_map, pred_atmosphere_light)

        r_loss = ref_loss(normal_image_rgb, pred_normal_image_rgb)
    
        # backward
        generator.zero_grad()
        total_loss.backward()
        opt_gen.step()

        refiner.zero_grad()
        r_loss.backward()
        opt_ref.step()

        curr_psnr += calculate_psnr(normal_image_rgb.detach().cpu().numpy() * 255.0, pred_normal_image_rgb.detach().cpu().numpy() * 255.0)
        avg_psnr = curr_psnr / (batch_idx + 1)

        train_writer.add_scalar("Generator loss", total_loss, global_step=train_step)
        train_writer.add_scalar("Refiner loss", r_loss, global_step=train_step)
        train_writer.add_scalar("PSNR", curr_psnr, global_step=train_step)
        
        # update tqdm loop
        loop.set_postfix(
            GenLoss=total_loss.item(),
            RefLoss=r_loss.item(),
            PSNR=avg_psnr,
        )
        train_step += 1

    return avg_psnr

def val_fn(generator, refiner, val_dataloader, val_writer, opt):

    generator.eval()
    refiner.eval()
    loop = tqdm(val_dataloader)
    curr_psnr = 0
    global val_step
    
    for batch_idx, (low_image_rgb, normal_image_rgb, normal_image_gray) in enumerate(loop):
        low_image_rgb = low_image_rgb.to(device=opt["device"])
        normal_image_rgb = normal_image_rgb.to(device=opt["device"])
        normal_image_gray = normal_image_gray.to(device=opt["device"])

        # get ground truth of trasmission map and atmospheric light
        orig_transmission_map, orig_atmosphere_light, orig_refined_image = get_refined_image(low_image_rgb, normal_image_rgb, opt)
        
        with torch.no_grad():
            # forward
            pred_normal_image_gray, pred_transmission_map, _, pred_refined_map = generator(low_image_rgb)
            pred_normal_image_rgb = refiner(pred_refined_map, pred_normal_image_gray)
        
        # originals
        low_img_grid = torchvision.utils.make_grid(low_image_rgb)
        orig_normal_img_grid = torchvision.utils.make_grid(normal_image_rgb)
        orig_gray_img_grid = torchvision.utils.make_grid(normal_image_gray)
        orig_tm_img_grid = torchvision.utils.make_grid(orig_transmission_map)
        orig_refined_map_grid = torchvision.utils.make_grid(orig_refined_image)

        val_writer.add_image("Low light image", low_img_grid, global_step=val_step)
        val_writer.add_image("Original normal light image", orig_normal_img_grid, global_step=val_step)
        val_writer.add_image("Original gray scale image", orig_gray_img_grid, global_step=val_step)
        val_writer.add_image("Original transmission map image", orig_tm_img_grid, global_step=val_step)
        val_writer.add_image("Original refined map image", orig_refined_map_grid, global_step=val_step)
        
        # preds
        pred_normal_img_grid = torchvision.utils.make_grid(pred_normal_image_rgb)
        pred_gray_img_grid = torchvision.utils.make_grid(pred_normal_image_gray)
        pred_tm_img_grid = torchvision.utils.make_grid(pred_transmission_map)
        pred_refined_map_grid = torchvision.utils.make_grid(pred_refined_map)
        
        val_writer.add_image("Predicted normal light image", pred_normal_img_grid, global_step=val_step)
        val_writer.add_image("Predicted gray scale image", pred_gray_img_grid, global_step=val_step)
        val_writer.add_image("Predicted transmission map image", pred_tm_img_grid, global_step=val_step)
        val_writer.add_image("Predicted refined map image", pred_refined_map_grid, global_step=val_step)

        
        curr_psnr += calculate_psnr(normal_image_rgb.detach().cpu().numpy() * 255.0, pred_normal_image_rgb.detach().cpu().numpy() * 255.0)
        avg_psnr = curr_psnr / (batch_idx + 1)

        # update tqdm loop
        loop.set_postfix(
            PSNR=avg_psnr,
        )
        val_step+=1
    return avg_psnr

def main():
    """
    main function for LLIE training
    """
    opt = parse_config()
    init_log(opt)
    logger = logging.getLogger(__name__)

    # for reproducibility
    seed_everything(42)

    # get dataloaders
    train_dataloader, val_dataloader = get_data(opt)
    logger.info(f"[Info] Found {train_dataloader.dataset.__len__()} number of images in training")
    logger.info(f"[Info] Found {val_dataloader.dataset.__len__()} number of images in validation")

    logger.info(f"[Info] Found {len(train_dataloader)} batches for training")
    logger.info(f"[Info] Found {len(val_dataloader)} batches for validation")

    # get model
    generator, refiner = get_model(opt["model"], device=opt["device"])
    initialize_weights(generator)
    initialize_weights(refiner)

    
    # tfboard writer
    train_writer = SummaryWriter(opt["path"]["train_logs"])
    val_writer = SummaryWriter(opt["path"]["val_logs"])

    # lossses and optimizers
    if opt["train"]["optim"]["type"] == "Adam":
        opt_gen = optim.Adam(generator.parameters(), lr=float(opt["train"]["optim"]["lr"]))
        opt_ref = optim.Adam(refiner.parameters(), lr=float(opt["train"]["optim"]["lr"]))
    
    elif opt["train"]["optim"]["type"] == "SGD":
        opt_gen = optim.SGD(generator.parameters(), lr=float(opt["train"]["optim"]["lr"]))
        opt_ref = optim.SGD(refiner.parameters(), lr=float(opt["train"]["optim"]["lr"]))
    
    gen_loss = GenLoss()
    ref_loss = RefineLoss()

    if opt["train"]["load_checkpoint"]:
        generator, refiner = load_checkpoint(
            generator, refiner, opt_gen, opt_ref, opt, logger
        )
    
    best_psnr = 0.0
    best_epoch = 0

    for epoch in range(opt["train"]["epochs"]):
        logger.info(f"[Info] Epoch : {epoch}/{opt['train']['epochs']}")
        train_psnr = train_fn(generator, refiner, train_dataloader, opt_gen, opt_ref, gen_loss, ref_loss, train_writer, opt)
        test_psnr = val_fn(generator, refiner, val_dataloader, val_writer, opt)
        logger.info(f'[Info] Training PSNR: {train_psnr}')
        logger.info(f'[Info] Testing PSNR: {test_psnr}')

        if test_psnr > best_psnr:
            best_psnr = test_psnr
            best_epoch = epoch
            logger.info(f"[Info] Saving checkpoint with best psnr of {test_psnr} at epoch {epoch}")
            save_checkpoint(generator, opt_gen, filename=opt["path"]["checkpoint_gen_network"])
            save_checkpoint(refiner, opt_ref, filename=opt["path"]["checkpoint_ref_network"])
        else:
            logger.info(f"[Info] Saved best checkpoint with best psnr of {best_psnr} at epoch {best_epoch}")

if __name__ == "__main__":
    main()