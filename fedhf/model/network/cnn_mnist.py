#!/usr/bin/env python
# -*- encoding: utf-8 -*-
""" 
@File    :   fedhf\model\cnn_cifar10.py 
@Time    :   2021-11-19 16:54:56 
@Author  :   Bingjie Yan 
@Email   :   bj.yan.pa@qq.com 
@License :   Apache License 2.0 
"""

import torch
import torch.nn as nn
from torch.nn.modules.activation import Softmax

from .base_model import BaseModel


class CNNMNIST(BaseModel):
    """
    Implentation of the cnn described in the fedasync
    """
    def __init__(self, args, model_time=0):
        super().__init__(args, model_time)

        self.num_classes = args.num_classes

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, self.num_classes),
        )

    def forward(self, x):
        return self.cnn(x)
