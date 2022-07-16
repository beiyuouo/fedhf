#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    :   fedhf\model\nn\unet.py
# @Time    :   2022-02-26 14:04:05

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...api import Config
from .base_model import BaseModel

"""
    U-Net: Convolutional Networks for Biomedical Image Segmentation
    Olaf Ronneberger, Philipp Fischer, Thomas Brox
    https://arxiv.org/abs/1505.04597

    Code References: https://github.com/milesial/Pytorch-UNet
    License: GNU General Public License v3.0
"""


class DoubleConv(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        hidden_channel=None,
        kernel_size=3,
        padding=1,
        bias=True,
    ):
        super(DoubleConv, self).__init__()
        if not hidden_channel:
            hidden_channel = out_channel
        self.double_conv = nn.Sequential(
            nn.Conv2d(
                in_channel,
                hidden_channel,
                kernel_size=kernel_size,
                padding=padding,
                bias=bias,
            ),
            nn.BatchNorm2d(hidden_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                hidden_channel,
                out_channel,
                kernel_size=kernel_size,
                padding=padding,
                bias=bias,
            ),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class DownConv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DownConv, self).__init__()
        self.down_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channel, out_channel),
        )

    def forward(self, x):
        return self.down_conv(x)


class UpConv(nn.Module):
    def __init__(self, in_channel, out_channel, bilinear=True):
        super(UpConv, self).__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channel, out_channel, in_channel // 2)
        else:
            self.up = nn.ConvTranspose2d(
                in_channel, in_channel // 2, kernel_size=2, stride=2
            )
            self.conv = DoubleConv(in_channel, out_channel)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(BaseModel):

    default_params = Config(unet={"bilinear": None, "n1": None})

    def __init__(self, args, model_time=None, model_version=0):
        super().__init__(args, model_time, model_version)
        self.add_default_args(self.default_params)
        print("UNet:", self.args.unet)

        self.input_c = self.args.input_c
        self.output_c = self.args.output_c
        self.bilinear = (
            self.args.unet.bilinear if self.args.unet.bilinear is not None else False
        )

        self.n1 = self.args.unet.n1 if self.args.unet.n1 is not None else 64
        self.filter = [self.n1, self.n1 * 2, self.n1 * 4, self.n1 * 8, self.n1 * 16]

        self.inc = DoubleConv(self.input_c, self.filter[0])
        self.down1 = DownConv(self.filter[0], self.filter[1])
        self.down2 = DownConv(self.filter[1], self.filter[2])
        self.down3 = DownConv(self.filter[2], self.filter[3])
        factor = 2 if self.bilinear else 1
        self.down4 = DownConv(self.filter[3], self.filter[4] // factor)
        self.up1 = UpConv(self.filter[4], self.filter[3] // factor, self.bilinear)
        self.up2 = UpConv(self.filter[3], self.filter[2] // factor, self.bilinear)
        self.up3 = UpConv(self.filter[2], self.filter[1] // factor, self.bilinear)
        self.up4 = UpConv(self.filter[1], self.filter[0], self.bilinear)
        self.outc = OutConv(self.filter[0], self.output_c)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        pred = self.outc(x)
        return pred


class UNetMini(BaseModel):
    default_params = Config(unet={"bilinear": None, "n1": None})

    def __init__(self, args, model_time=None, model_version=0):
        super().__init__(args, model_time, model_version)
        self.add_default_args(self.default_params)

        self.input_c = self.args.input_c
        self.output_c = self.args.output_c
        self.bilinear = (
            self.args.unet.bilinear if self.args.unet.bilinear is not None else False
        )

        self.n1 = self.args.unet.n1 if self.args.unet.n1 is not None else 64
        self.filter = [self.n1, self.n1 * 2, self.n1 * 4]

        factor = 2 if self.bilinear else 1

        self.inc = DoubleConv(self.input_c, self.filter[0])
        self.down1 = DownConv(self.filter[0], self.filter[1])
        self.down2 = DownConv(self.filter[1], self.filter[2] // factor)
        self.up1 = UpConv(self.filter[2], self.filter[1] // factor, self.bilinear)
        self.up2 = UpConv(self.filter[1], self.filter[0], self.bilinear)
        self.outc = OutConv(self.filter[0], self.output_c)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x = self.up1(x3, x2)
        x = self.up2(x, x1)
        pred = self.outc(x)
        return pred
