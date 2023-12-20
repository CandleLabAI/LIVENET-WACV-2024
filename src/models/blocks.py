# ------------------------------------------------------------------------
# Copyright (c) 2023 CandleLabAI. All Rights Reserved.
# ------------------------------------------------------------------------

"""
Network blocks for Livenet
"""

# imports
import torch.nn as nn
import torch

class SEAttention(nn.Module):   #it gives channel attention
    def __init__(self, in_channels, reduced_dim=16):  #input_shape ---> output_shape
        super(SEAttention, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), # C x H x W -> C x 1 x 1
            nn.Conv2d(in_channels, reduced_dim, 1),
            nn.GELU(),
            nn.Conv2d(reduced_dim, in_channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.se(x)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(ResidualBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=self.kernel_size, stride=stride, padding=(self.kernel_size - 1) // 2, bias=False),
            nn.InstanceNorm2d(self.out_channels),
            nn.GELU(),
            nn.Conv2d(in_channels=self.out_channels, out_channels=self.out_channels, kernel_size=self.kernel_size, stride=1, padding=(self.kernel_size - 1) // 2, bias=False),
            nn.InstanceNorm2d(self.out_channels),
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, padding=0, stride=stride, bias=False),
            nn.InstanceNorm2d(self.out_channels),
            SEAttention(in_channels=self.out_channels),
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.backbone(x) + self.shortcut(x))

class LSDB(nn.Module):
    def __init__(self, in_nc, out_nc):
        super(LSDB, self).__init__()
        self.shallow = nn.Conv2d(in_nc, out_nc, 3, 1, 1)
        self.fu = nn.Sequential(
            nn.Conv2d(out_nc, out_nc*2, 1, 1, 0),
            nn.GELU(),  
            nn.Conv2d(out_nc*2, out_nc*2, 1, 1, 0),
            nn.GELU())
        self.fv = nn.Sequential(
            nn.Conv2d(out_nc, out_nc, 1, 1, 0),
            nn.GELU(),
            nn.Conv2d(out_nc, out_nc, 1, 1, 0),
            nn.GELU())

    def forward(self, x):
        x = self.shallow(x)
        fu = self.fu(x)
        u = fu.view(fu.shape[0], fu.shape[1], -1)
        uT = u.permute(0, 2, 1)
        fv = self.fv(x)
        f = fv.view(fv.shape[0], fv.shape[1], -1)
        f = f.permute(0, 2, 1)
        v = torch.matmul(u, f)
        F = torch.matmul(uT, v)
        F = F.view(F.shape[0], x.shape[2], x.shape[3], x.shape[1])
        F = F.permute(0, 3, 1, 2)

        F = x + F
        return F

class GrayEncoder(nn.Module):
    def __init__(self, in_nc):
        super(GrayEncoder, self).__init__()

        self.block1 = nn.Sequential(
                                    ResidualBlock(in_channels=in_nc, out_channels=16, kernel_size=3, stride=1), # 512
                                    LSDB(in_nc = 16, out_nc = 16),
                                    )

        self.block2 = nn.Sequential(
                                    ResidualBlock(in_channels=16, out_channels=32, kernel_size=3, stride=2), # 256
                                    LSDB(in_nc = 32, out_nc =  32),
                                    )

        self.block3 = nn.Sequential(
                                    ResidualBlock(in_channels=32, out_channels=64, kernel_size=3, stride=2), # 128
                                    LSDB(in_nc = 64, out_nc = 64),
                                    )
        self.block4 = nn.Sequential(
                                    ResidualBlock(in_channels=64, out_channels=128, kernel_size=3, stride=2), # 64
                                    LSDB(in_nc = 128, out_nc = 128),
                                    )
        
        self.block5 = nn.Sequential(
                                    ResidualBlock(in_channels=128, out_channels=128, kernel_size=3, stride=2), # 32
                                    LSDB(in_nc = 128, out_nc = 128),
                                    ) 
        self.block6 = nn.Sequential(
                                    ResidualBlock(in_channels=128, out_channels=256, kernel_size=3, stride=2), # 16
                                    LSDB(in_nc = 256, out_nc = 256),
                                    )

    def forward(self, x):
        x1 = self.block1(x)
        x2 = self.block2(x1)
        x3 = self.block3(x2)
        x4 = self.block4(x3)
        x5 = self.block5(x4)
        x6 = self.block6(x5)
        return x2, x3, x4, x5, x6

class ResidualEncoder(nn.Module):
    def __init__(self, in_nc):
        super(ResidualEncoder, self).__init__()

        self.block1 = ResidualBlock(in_channels=in_nc, out_channels=16, kernel_size=3, stride=1) # 512

        self.block2 = ResidualBlock(in_channels=16, out_channels=32, kernel_size=3, stride=2) # 256

        self.block3 = ResidualBlock(in_channels=32, out_channels=64, kernel_size=3, stride=2) # 128
        self.block4 = ResidualBlock(in_channels=64, out_channels=128, kernel_size=3, stride=2) # 64

        self.block5 = ResidualBlock(in_channels=128, out_channels=128, kernel_size=3, stride=2) # 32 
        self.block6 = ResidualBlock(in_channels=128, out_channels=256, kernel_size=3, stride=2) # 16

    def forward(self, x):
        x1 = self.block1(x)
        x2 = self.block2(x1)
        x3 = self.block3(x2)
        x4 = self.block4(x3)
        x5 = self.block5(x4)
        x6 = self.block6(x5)
        return x2, x3, x4, x5, x6

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()

        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        self.ResidualBlock = ResidualBlock(in_channels=out_channels*2, out_channels=out_channels, kernel_size=3, stride=1)

    def forward(self, feature, skip_f):
        x = self.upsample(feature)
        x = torch.cat([x, skip_f], axis=1)
        x = self.ResidualBlock(x)
        return x

class ResidualDecoder(nn.Module):
    def __init__(self, out_nc):
        super(ResidualDecoder, self).__init__()

        self.block1 = DecoderBlock(in_channels=256, out_channels=128)
        self.block2 = DecoderBlock(in_channels=128, out_channels=128)
        self.block3 = DecoderBlock(in_channels=128, out_channels=64)
        self.block4 = DecoderBlock(in_channels=64, out_channels=32)
        self.block5 = nn.Sequential(nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
                                    nn.Conv2d(16, out_nc, 3, 1, 1),
                                    nn.Tanh()
                                    )

    def forward(self, x5, x4, x3, x2, x1):
        x = self.block1(x5, x4)
        x = self.block2(x, x3)
        x = self.block3(x, x2)
        x = self.block4(x, x1)
        x = (self.block5(x) + 1) / 2
        return x

class SFT_layer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SFT_layer, self).__init__()
        self.SFT_scale = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 1, 1, 0, bias=False),
        )
        self.SFT_shift = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 1, 1, 0, bias=False),
        )

    def forward(self, coarse, gray):
        '''
        :param x: degradation representation: B * C
        :param inter: degradation intermediate representation map: B * C * H * W
        '''
        scale = self.SFT_scale(gray)
        shift = self.SFT_shift(gray)
        return coarse * (scale + 1) + shift

class RefineBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(RefineBlock, self).__init__()

        self.block1 = ResidualBlock(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride)
        self.block2 = ResidualBlock(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride)
        self.sft = SFT_layer(in_channels=out_channels, out_channels=out_channels)

    def forward(self, x1, x2):
        x1 = self.block1(x1)
        x2 = self.block2(x2)
        x = self.sft(x1, x2)
        return x1, x2, x

