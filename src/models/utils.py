# ------------------------------------------------------------------------
# Copyright (c) 2023 CandleLabAI. All Rights Reserved.
# ------------------------------------------------------------------------
"""
basic helper functions
"""

from torch import Tensor
import numpy as np
import torch
import math

def get_final_image(input_image, atmosphere_light, refined_tranmission_map, tmin):
    """
    input_image: (1x3x256x256) = NCHW
    atmosphere_light: (1x3) = NC
    refined_tranmission_map: (1x1x256x256) = NCHW
    """
    refined_tranmission_map_broadcasted = torch.broadcast_to(refined_tranmission_map, (refined_tranmission_map.shape[0], 3, refined_tranmission_map.shape[2], refined_tranmission_map.shape[3])).to("cuda")
    refined_tranmission_map_broadcasted = refined_tranmission_map_broadcasted.permute(0, 2, 3, 1)
    input_image = input_image.permute(0, 2, 3, 1)
    refined_image = torch.empty(size=input_image.shape, dtype=input_image.dtype, device=input_image.device)
    for batch in range(input_image.shape[0]):
        refined_image[batch, :, :, :] = (input_image[batch] - (1.0 - refined_tranmission_map_broadcasted[batch]) * atmosphere_light[batch]) / (torch.where(refined_tranmission_map_broadcasted[batch] < tmin, tmin, refined_tranmission_map_broadcasted[batch]))
        refined_image[batch] = (refined_image[batch] - torch.min(refined_image[batch])) / (torch.max(refined_image[batch]) - torch.min(refined_image[batch]))
    refined_image = refined_image.permute(0, 3, 1, 2)
    return refined_image

def get_corrected_transmission_map(input_image, atmosphere_light, dark_channel_prior, bright_channel_prior, initial_transmission_map, omega, alpha, channel_prior_kernel):
    """
    input_image: (1x3x256x256) = NCHW
    atmosphere_light: (1x3) = NC
    dark_channel_prior: (1x1x256x256) = NCHW
    bright_channel_prior: (1x1x256x256) = NCHW
    initial_transmission_map: (1x1x256x256) = NCHW
    """
    img = (1 - input_image) / (1 - atmosphere_light[:, :, None, None] + 1e-6)
    _, dark_channel_transmissionmap = get_illumination_channel(img, channel_prior_kernel)
    dark_channel_transmissionmap = 1.0 - omega * dark_channel_transmissionmap
    corrected_transmission_map = initial_transmission_map
    difference_channel_prior = bright_channel_prior - dark_channel_prior
    indices = difference_channel_prior < alpha
    corrected_transmission_map[indices] = dark_channel_transmissionmap[indices] * initial_transmission_map[indices]
    return corrected_transmission_map

def get_initial_transmission_map(atmosphere_light, bright_channel_prior):
    """
    atmosphere_light: (1x3) = NC
    bright_channel_prior: (1x1x256x256) = NCHW
    initial_transmission_map: (1x1x256x256) = NCHW
    """
    initial_transmission_map = torch.empty(size=bright_channel_prior.shape, dtype=bright_channel_prior.dtype, device=bright_channel_prior.device)
    for batch in range(bright_channel_prior.shape[0]):
        initial_transmission_map[batch] = (bright_channel_prior[batch] - torch.max(atmosphere_light[batch])) / (1.0 - torch.max(atmosphere_light[batch]))
        initial_transmission_map[batch] = (initial_transmission_map[batch] - torch.min(initial_transmission_map[batch])) / (torch.max(initial_transmission_map[batch]) - torch.min(initial_transmission_map[batch]))
    return initial_transmission_map

def get_global_atmosphere_light(input_image, bright_channel_prior, probability=0.001):
    """
    input_image: (1x3x256x256) = NCHW
    bright_channel_prior: (1x1x256x256) = NCHW
    atmosphere_light: (1x3) = NC
    """
    flattened_image = input_image.view(input_image.shape[0], input_image.shape[1], input_image.shape[2] * input_image.shape[3])
    flattened_bright_channel_prior = bright_channel_prior.view(bright_channel_prior.shape[0], bright_channel_prior.shape[2]*bright_channel_prior.shape[3])
    index = torch.argsort(flattened_bright_channel_prior, dim=-1, descending=False)[:, :int(input_image.shape[2] * input_image.shape[3] * probability)]
    atmosphere_light = torch.zeros((input_image.shape[0], 3), device="cuda")
    for i in range(input_image.shape[0]):
        atmosphere_light[i] = flattened_image[i, :, index].mean(axis=(1, 2))
    return atmosphere_light

def get_illumination_channel(input_image, channel_prior_kernel):
    """
    input_image: (1x3x256x256) = NCHW
    dark_channel_prior: (1x1x256x256) = NCHW
    bright_channel_prior: (1x1x256x256) = NCHW
    """
    maxpool = torch.nn.MaxPool3d((3, channel_prior_kernel, channel_prior_kernel), stride=(1, 1, 1), padding=(0, channel_prior_kernel // 2, channel_prior_kernel // 2))
    bright_channel_prior = maxpool(input_image)
    dark_channel_prior = maxpool(0.0 - input_image)
    
    return -dark_channel_prior, bright_channel_prior

def get_refined_image(low_image_rgb, normal_image_rgb, opt):
    low_image_rgb = low_image_rgb.to(opt["device"])
    normal_image_rgb = normal_image_rgb.to(opt["device"]) 
    input_image = (low_image_rgb + normal_image_rgb) / 2.0
    dark_channel_prior, bright_channel_prior = get_illumination_channel(input_image, opt["model"]["channel_prior_kernel"])
    atmosphere_light = get_global_atmosphere_light(low_image_rgb, bright_channel_prior)
    initial_transmission_map = get_initial_transmission_map(atmosphere_light, bright_channel_prior)
    transmission_map = get_corrected_transmission_map(low_image_rgb, atmosphere_light, dark_channel_prior, bright_channel_prior, initial_transmission_map, opt["model"]["omega"], opt["model"]["alpha"], opt["model"]["channel_prior_kernel"])
    refined_image = get_final_image(low_image_rgb, atmosphere_light, transmission_map, opt["model"]["tmin"])
    return transmission_map, atmosphere_light, refined_image

def rgb_to_ycbcr(image):
    r"""Convert an RGB image to YCbCr.

    Args:
        image (torch.Tensor): RGB Image to be converted to YCbCr.

    Returns:
        torch.Tensor: YCbCr version of the image.
    """

    if not torch.is_tensor(image):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(image)))

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError("Input size must have a shape of (*, 3, H, W). Got {}"
                         .format(image.shape))

    r: torch.Tensor = image[..., 0, :, :]
    g: torch.Tensor = image[..., 1, :, :]
    b: torch.Tensor = image[..., 2, :, :]

    delta = .5
    y: torch.Tensor = .299 * r + .587 * g + .114 * b
    cb: torch.Tensor = (b - y) * .564 + delta
    cr: torch.Tensor = (r - y) * .713 + delta
    return torch.stack((y, cb, cr), -3)

def ycbcr_to_rgb(image):
    r"""Convert an YCbCr image to RGB.

    The image data is assumed to be in the range of (0, 1).

    Args:
        image: YCbCr Image to be converted to RGB with shape :math:`(*, 3, H, W)`.

    Returns:
        RGB version of the image with shape :math:`(*, 3, H, W)`.

    Examples:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = ycbcr_to_rgb(input)  # 2x3x4x5
    """
    if not isinstance(image, Tensor):
        raise TypeError(f"Input type is not a Tensor. Got {type(image)}")

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of (*, 3, H, W). Got {image.shape}")

    y = image[..., 0, :, :]
    cb = image[..., 1, :, :]
    cr = image[..., 2, :, :]

    delta: float = 0.5
    cb_shifted = cb - delta
    cr_shifted = cr - delta

    r = y + 1.403 * cr_shifted
    g = y - 0.714 * cr_shifted - 0.344 * cb_shifted
    b = y + 1.773 * cb_shifted
    return torch.stack([r, g, b], -3)

def combine_YCbCr_and_RGB(rgb_image, ycbcr_image):
    rgb_to_ycbcr_image = rgb_to_ycbcr(rgb_image)
    rgb_to_ycbcr_image[:, 0, :, :] = ycbcr_image[:, 0, :, :]
    rgb_image = ycbcr_to_rgb(rgb_to_ycbcr_image)
    return rgb_image, ycbcr_image

def initialize_weights(model):
    # Initializes weights according to the DCGAN paper
    for m in model.modules():
        if isinstance(m, (torch.nn.Conv2d, torch.nn.ConvTranspose2d)):
            torch.nn.init.normal_(m.weight.data, 0.0, 0.02)

def save_checkpoint(model, optimizer, filename="my_checkpoint.pth"):
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(gen, ref, opt_gen, opt_ref, opt, logger):
    logger.info("[Info] Loading Generator checkpoints...")
    checkpoint = torch.load(opt["path"]["checkpoint_gen_network"], map_location=opt["device"])
    gen.load_state_dict(checkpoint["state_dict"])
    opt_gen.load_state_dict(checkpoint["optimizer"])
    logger.info("[Info] Generator Weights loaded successfully")
    
    logger.info("[Info] Loading Refiner checkpoints...")
    checkpoint = torch.load(opt["path"]["checkpoint_ref_network"], map_location=opt["device"])
    ref.load_state_dict(checkpoint["state_dict"])
    opt_ref.load_state_dict(checkpoint["optimizer"])
    logger.info("[Info] Refiner Weights loaded successfully")

    return gen, ref

def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))