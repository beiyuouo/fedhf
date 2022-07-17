#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    :   fedhf\model\nn\__init__.py
# @Time    :   2022-05-03 16:06:57
# @Author  :   Bingjie Yan
# @Email   :   bj.yan.pa@qq.com
# @License :   Apache License 2.0

from .base_model import BaseModel
from .cv import *

model_factory = {
    "resnet": ResNet,
    "resnet18": ResNet,
    "resnet_mnist": ResNetMNIST,
    "mlp": MLP,
    "alexnet_cifar10": AlexNetCIFAR10,
    "cnn_cifar10": CNN2CIFAR10,
    "cnn2_cifar10": CNN2CIFAR10,
    "cnn4_cifar10": CNN4CIFAR10,
    "cnn_mnist": CNNMNIST,
    "densenet": DenseNet,
    "unet": UNet,
    "unet_mini": UNetMini,
}


def build_model(model_name: str):
    if model_name not in model_factory.keys():
        raise ValueError(f"Unknown model name: {model_name}")
    model = model_factory[model_name]
    return model
