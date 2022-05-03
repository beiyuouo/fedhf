#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    :   fedhf\model\criterion\__init__.py
# @Time    :   2022-05-03 16:06:51
# @Author  :   Bingjie Yan
# @Email   :   bj.yan.pa@qq.com
# @License :   Apache License 2.0

import torch
import torch.nn as nn
import torch.optim as optim

criterion_factory = {
    'l1': nn.L1Loss,
    'mse': nn.MSELoss,
    'ce': nn.CrossEntropyLoss,
    'bce': nn.BCELoss,
}


def build_criterion(criter_name: str):
    if criter_name not in criterion_factory.keys():
        raise ValueError(f'Unknown criterion name: {criter_name}')

    cirter = criterion_factory[criter_name]
    return cirter
