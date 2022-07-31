#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    :   fedhf\model\nn\resnet.py
# @Time    :   2022-05-03 16:07:31
# @Author  :   Bingjie Yan
# @Email   :   bj.yan.pa@qq.com
# @License :   Apache License 2.0

import torch
import torch.nn as nn

from torchvision import models

from ..base_model import BaseModel

"""
    Deep Residual Learning for Image Recognition
    Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    https://arxiv.org/abs/1512.03385v1
"""


class ResNet(BaseModel):
    def __init__(self, args, **kwargs):
        super().__init__(args, **kwargs)
        self.net = models.resnet18(pretrained=True)
        self.net.fc = nn.Linear(512, args.num_classes)

    def forward(self, x):
        return self.net(x)
