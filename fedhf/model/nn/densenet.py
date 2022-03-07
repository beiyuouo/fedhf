#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    :   fedhf\model\nn\densenet.py
# @Time    :   2022-02-26 13:55:26
# @Author  :   Bingjie Yan
# @Email   :   bj.yan.pa@qq.com
# @License :   Apache License 2.0

import torch
import torch.nn as nn
from torchvision import models

from .base_model import BaseModel


class DenseNet(BaseModel):
    def __init__(self, args, model_time=None, model_version=0):
        super().__init__(args, model_time, model_version)
        self.net = models.densenet121(pretrained=True)
        self.num_classes = args.num_classes
        self.net.classifier = nn.Linear(1024, args.num_classes)

    def forward(self, x):
        return self.net(x)