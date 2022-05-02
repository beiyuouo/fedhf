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

    def __init__(self, length, data_shape):
        self.length = length
        self.data_shape = data_shape

        self.data = torch.randn(length, *data_shape)
        self.label = torch.randint(0, 10, (length,))

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return self.length