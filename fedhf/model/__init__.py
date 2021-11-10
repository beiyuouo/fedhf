#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   fedhf\model\__init__.py
@Time    :   2021-10-28 15:02:58
@Author  :   Bingjie Yan
@Email   :   bj.yan.pa@qq.com
@License :   Apache License 2.0
"""

__all__ = ['build_model', 'build_optimizer', 'build_loss']

import torch
import torch.nn as nn
import torch.optim as optim
from .resnet import ResNet
from .mlp import MLP
from .cnn import CNN

model_factory = {
    'resnet': ResNet,
    'resnet18': ResNet,
    'mlp': MLP,
    'cnn': CNN
}


def build_model(model_name: str):
    if model_name not in model_factory.keys():
        raise ValueError(f'Unknown model name: {model_name}')

    model = model_factory[model_name]

    return model


optimizer_factory = {
    'sgd': optim.SGD,
    'adam': optim.Adam,
    'adagrad': optim.Adagrad
}


def build_optimizer(optim_name: str):
    if optim_name not in optimizer_factory.keys():
        raise ValueError(f'Unknown optimizer name: {optim_name}')

    optimizer = optimizer_factory[optim_name]

    return optimizer


criterion_factory = {
    'l1': nn.L1Loss,
    'mse': nn.MSELoss,
    'ce': nn.CrossEntropyLoss,
    'bce': nn.BCELoss
}


def build_criterion(criter_name: str):
    if criter_name not in criterion_factory.keys():
        raise ValueError(f'Unknown criterion name: {criter_name}')

    cirter = criterion_factory[criter_name]
    return cirter
