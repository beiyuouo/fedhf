#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    :   fedhf\model\nn\cnn_cifar10.py
# @Time    :   2022-05-03 16:07:12
# @Author  :   Bingjie Yan
# @Email   :   bj.yan.pa@qq.com
# @License :   Apache License 2.0

import torch
import torch.nn as nn
from torch.nn.modules.activation import Softmax

from ..base_model import BaseModel


class CNN4CIFAR10(BaseModel):
    """
    Implentation of the cnn described in the fedasync
    """

    def __init__(self, args, model_time=None, model_version=0):
        super().__init__(args, model_time, model_version)

        self.num_classes = args.num_classes

        self.net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 512),
            nn.ReLU(0.1),
            nn.Dropout(0.25),
            nn.Linear(512, self.num_classes),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        return self.net(x)


class CNN2CIFAR10(BaseModel):
    """
    Implentation of the cnn described in the fedasync
    """

    def __init__(self, args, model_time=None, model_version=0):
        super().__init__(args, model_time, model_version)

        self.num_classes = args.num_classes

        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, self.num_classes),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        return self.net(x)
