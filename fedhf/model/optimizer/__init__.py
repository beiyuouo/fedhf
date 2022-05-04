#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    :   fedhf\model\optimizer\__init__.py
# @Time    :   2022-05-03 16:07:47
# @Author  :   Bingjie Yan
# @Email   :   bj.yan.pa@qq.com
# @License :   Apache License 2.0

import torch
import torch.nn as nn
import torch.optim as optim

optimizer_factory = {
    'sgd': optim.SGD,
    'adam': optim.Adam,
    'adagrad': optim.Adagrad,
}


def build_optimizer(optim_name: str):
    if optim_name not in optimizer_factory.keys():
        raise ValueError(f'Unknown optimizer name: {optim_name}')

    optimizer = optimizer_factory[optim_name]

    return optimizer
