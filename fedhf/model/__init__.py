#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   fedhf\model\__init__.py
@Time    :   2021-10-28 15:02:58
@Author  :   Bingjie Yan
@Email   :   bj.yan.pa@qq.com
@License :   Apache License 2.0
"""

__all__ = [
    "build_model", "build_optimizer", "build_criterion", "criterion_factory",
    "optimizer_factory", "model_factory", "build_lr_scheduler", "lr_schedule_factory"
]

from .criterion import build_criterion, criterion_factory
from .nn import build_model, model_factory
from .optimizer import build_optimizer, optimizer_factory
from .scheduler import build_lr_scheduler, lr_schedule_factory

__all__ += [
    "ResNet", "ResNetMNIST", "MLP", "AlexNetCIFAR10", "CNN2CIFAR10", "CNN4CIFAR10", "CNNMNIST",
    "DenseNet", "UNet", "BaseModel"
]

from .nn import ResNet, ResNetMNIST, MLP, AlexNetCIFAR10, CNN2CIFAR10, CNN4CIFAR10, CNNMNIST, DenseNet, UNet, BaseModel