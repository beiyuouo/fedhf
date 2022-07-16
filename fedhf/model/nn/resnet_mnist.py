#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    :   fedhf\model\nn\resnet_mnist.py
# @Time    :   2022-05-03 16:07:27
# @Author  :   Bingjie Yan
# @Email   :   bj.yan.pa@qq.com
# @License :   Apache License 2.0

import torch
import torch.nn as nn

from torchvision import models

from .base_model import BaseModel


class ResNetMNIST(BaseModel):
    def __init__(self, args, model_time=None, model_version=0):
        super().__init__(args, model_time, model_version)
        self.input_size = (args.input_c, args.image_size, args.image_size)
        self.num_classes = args.num_classes
        self.conv = nn.Conv2d(1, 3, kernel_size=1)
        self.net = models.resnet18(pretrained=args.model_pretrained)
        self.net.fc = nn.Linear(512, args.num_classes)

    def forward(self, x):
        return self.net(self.conv(x))
