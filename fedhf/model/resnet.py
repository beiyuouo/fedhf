#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   fedhf\model\resnet.py
@Time    :   2021-10-28 15:41:49
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
        self.cnn = models.resnet18(pretrained=False)
        self.cnn.fc = nn.Linear(512, args.num_classes)

    def forward(self, x):
        return self.cnn(x)
