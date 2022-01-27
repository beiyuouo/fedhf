#!/usr/bin/env python
# -*- encoding: utf-8 -*-
""" 
@File    :   fedhf\model\network\__init__.py 
@Time    :   2022-01-24 11:49:23 
@Author  :   Bingjie Yan 
@Email   :   bj.yan.pa@qq.com 
@License :   Apache License 2.0 
"""

from .resnet import ResNet
from .mlp import MLP
from .alexnet_cifar10 import AlexNetCIFAR10
from .cnn_cifar10 import CNN2CIFAR10, CNN4CIFAR10
from .cnn_mnist import CNNMNIST

model_factory = {
    'resnet': ResNet,
    'resnet18': ResNet,
    'mlp': MLP,
    'alexnet_cifar10': AlexNetCIFAR10,
    'cnn_cifar10': CNN2CIFAR10,
    'cnn2_cifar10': CNN2CIFAR10,
    'cnn4_cifar10': CNN4CIFAR10,
    'cnn_mnist': CNNMNIST,
}


def build_model(model_name: str):
    if model_name not in model_factory.keys():
        raise ValueError(f'Unknown model name: {model_name}')

    model = model_factory[model_name]

    return model
