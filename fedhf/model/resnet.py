#!/usr/bin/env python
# -*- encoding: utf-8 -*-
""" 
@File    :   fedhf\model\resnet.py 
@Time    :   2021-11-11 12:11:41 
@Author  :   Bingjie Yan 
@Email   :   bj.yan.pa@qq.com 
@License :   Apache License 2.0 
"""

import torch
import torch.nn as nn

from torchvision import models


class ResNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.input_size = (args.input_c, args.image_size, args.image_size)
        self.num_classes = args.num_classes

        self.conv = nn.Conv2d(1, 3, kernel_size=1)
        self.cnn = models.resnet18(pretrained=args.model_pretrained)
        self.cnn.fc = nn.Linear(512, args.num_classes)

    def forward(self, x):
        return self.cnn(self.conv(x))
