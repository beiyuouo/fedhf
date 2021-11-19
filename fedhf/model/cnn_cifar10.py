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


class CNNCIFAR10(BaseModel):
    """
    Implentation of the cnn described in the fedasync
    """
    def __init__(self, args, model_time=0):
        super().__init__(args, model_time)

        self.num_classes = args.num_classes

        self.cnn = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3, padding=1),
                                 nn.ReLU(), nn.BatchNorm2d(64),
                                 nn.Conv2d(64, 64, kernel_size=3, padding=1),
                                 nn.ReLU(), nn.BatchNorm2d(64),
                                 nn.MaxPool2d(kernel_size=2), nn.Dropout(0.25),
                                 nn.Conv2d(64, 128, kernel_size=3, padding=1),
                                 nn.ReLU(), nn.BatchNorm2d(128),
                                 nn.MaxPool2d(kernel_size=2), nn.Dropout(0.25),
                                 nn.Flatten(), nn.Linear(128 * 6 * 6, 512),
                                 nn.ReLU(), nn.Dropout(0.5),
                                 nn.Linear(512, 10), nn.Softmax())

    def forward(self, x):
        return self.cnn(x)
