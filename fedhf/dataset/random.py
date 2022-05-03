#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    :   fedhf\dataset\random.py
# @Time    :   2022-05-03 03:33:42
# @Author  :   Bingjie Yan
# @Email   :   bj.yan.pa@qq.com
# @License :   Apache License 2.0

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


class RandomDataset(Dataset):
    """
    Generate random data.
    """

    def __init__(self, args, length, data_shape, num_classes=None):
        self.args = args
        self.length = length
        self.data_shape = data_shape
        self.num_classes = num_classes if num_classes is not None else args.num_classes
        self.num_classes = self.num_classes if self.num_classes is not None else 10

        self.data = torch.randn(length, *data_shape)
        self.label = torch.randint(0, self.num_classes, (length,))

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return self.length