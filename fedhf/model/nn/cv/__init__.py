#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    :   fedhf\model\nn\cv\__init__.py
# @Time    :   2022-07-16 23:32:13
# @Author  :   Bingjie Yan
# @Email   :   bj.yan.pa@qq.com
# @License :   Apache License 2.0

from .resnet import ResNet
from .resnet_mnist import ResNetMNIST
from .mlp import MLPMNIST
from .alexnet_cifar10 import AlexNetCIFAR10
from .cnn_cifar10 import CNN2CIFAR10, CNN4CIFAR10
from .cnn_mnist import CNNMNIST
from .densenet import DenseNet
from .unet import UNet, UNetMini


cv_model_factory = {
    "resnet": ResNet,
    "resnet18": ResNet,
    "resnet_mnist": ResNetMNIST,
    "mlp_mnist": MLPMNIST,
    "alexnet_cifar10": AlexNetCIFAR10,
    "cnn_cifar10": CNN2CIFAR10,
    "cnn2_cifar10": CNN2CIFAR10,
    "cnn4_cifar10": CNN4CIFAR10,
    "cnn_mnist": CNNMNIST,
    "densenet": DenseNet,
    "unet": UNet,
    "unet_mini": UNetMini,
}
