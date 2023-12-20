# ------------------------------------------------------------------------
# Copyright (c) 2023 CandleLabAI. All Rights Reserved.
# ------------------------------------------------------------------------

"""
Network definition for Livenet
"""

# imports
from .blocks import GrayEncoder, ResidualEncoder, ResidualDecoder, RefineBlock, DecoderBlock
from .utils import get_final_image, combine_YCbCr_and_RGB
import torch.nn.functional as F
import torch.nn as nn
import torch



class Generator(torch.nn.Module):
    def __init__(self, opt):
        super(Generator, self).__init__()
        self.opt = opt
        self.gray_encoder = GrayEncoder(in_nc=opt["gray"]["in_nc"])
        self.residual_encoder = ResidualEncoder(in_nc=opt["tranmission_map"]["in_nc"])
        self.gray_decoder = ResidualDecoder(out_nc=opt["gray"]["out_nc"])
        self.tm_decoder = ResidualDecoder(out_nc=opt["tranmission_map"]["out_nc"])
        self.atmos_decoder = ResidualDecoder(out_nc=opt["atmospheric_light"]["out_nc"])

    def forward(self, input_tensor):
        gfm_1, gfm_2, gfm_3, gfm_4, gfm_5 = self.gray_encoder(input_tensor)
        fm_1, fm_2, fm_3, fm_4, fm_5 = self.residual_encoder(input_tensor)
        gray = self.gray_decoder(gfm_5, gfm_4, gfm_3, gfm_2, gfm_1)
        tm = self.tm_decoder(fm_5, fm_4, fm_3, fm_2, fm_1)
        atmos = self.atmos_decoder(fm_5, fm_4, fm_3, fm_2, fm_1)
        atmos = F.avg_pool2d(atmos, (atmos.shape[2], atmos.shape[3]))
        atmos = atmos.view(atmos.shape[0], -1)
        coarsemap = get_final_image(input_tensor.detach(), atmos.detach(), tm.detach(), self.opt["tmin"])
        coarsemap, gray = combine_YCbCr_and_RGB(coarsemap, gray)
        return gray, tm, atmos, coarsemap

class Refiner(torch.nn.Module):
    def __init__(self, opt):
        super(Refiner, self).__init__()

        self.block1 = RefineBlock(in_channels=opt["refiner"]["in_nc"], out_channels=16, kernel_size=3, stride=1)

        self.block2 = RefineBlock(in_channels=16, out_channels=32, kernel_size=3, stride=2)
        self.block3 = RefineBlock(in_channels=32, out_channels=64, kernel_size=3, stride=2)

        self.block4 = RefineBlock(in_channels=64, out_channels=128, kernel_size=3, stride=2)
        self.block5 = RefineBlock(in_channels=128, out_channels=128, kernel_size=3, stride=2)
        
        self.block6 = RefineBlock(in_channels=128, out_channels=256, kernel_size=3, stride=2)

        self.block7 = DecoderBlock(in_channels=256, out_channels=128)
        self.block8 = DecoderBlock(in_channels=128, out_channels=128)
        self.block9 = DecoderBlock(in_channels=128, out_channels=64)
        self.block10 = DecoderBlock(in_channels=64, out_channels=32)
        self.block11 = nn.Sequential(nn.ConvTranspose2d(32, 8, kernel_size=4, stride=2, padding=1),
                                    nn.Conv2d(8, opt["refiner"]["out_nc"], 3, 1, 1),
                                    nn.Tanh()
                                    )

    def forward(self, coarsemap, gray):
        gray = gray.repeat(1, 3, 1, 1)

        coarse1, gray1, sft1 = self.block1(coarsemap, gray)
        coarse2, gray2, sft2 = self.block2(coarse1, gray1)
        coarse3, gray3, sft3 = self.block3(coarse2, gray2)
        coarse4, gray4, sft4 = self.block4(coarse3, gray3)
        coarse5, gray5, sft5 = self.block5(coarse4, gray4)
        coarse6, gray6, sft6 = self.block6(coarse5, gray5)

        out = self.block7(sft6, sft5)
        out = self.block8(out, sft4)
        out = self.block9(out, sft3)
        out = self.block10(out, sft2)
        out = (self.block11(out) + 1) / 2

        return out

def get_model(opt, device):
    gen = Generator(opt).to(device)
    ref = Refiner(opt).to(device)
    return gen, ref