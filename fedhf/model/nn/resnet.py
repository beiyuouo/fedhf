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

from .base_model import BaseModel
"""
    [1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun.
        Deep Residual Learning for Image Recognition
        https://arxiv.org/abs/1512.03385v1
"""


class ResNet(BaseModel):
    def __init__(self, args, model_time=None, model_version=0):
        super().__init__(args, model_time, model_version)
        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(512, args.num_classes)

    def forward(self, x):
        return self.model(x)
