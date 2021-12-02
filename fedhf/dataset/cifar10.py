#!/usr/bin/env python
# -*- encoding: utf-8 -*-
""" 
@File    :   fedhf\dataset\cifar10.py 
@Time    :   2021-11-17 16:11:03 
@Author  :   Bingjie Yan 
@Email   :   bj.yan.pa@qq.com 
@License :   Apache License 2.0 
"""

from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
import torch


class CIFAR10Dataset(object):
    def __init__(self, args) -> None:
        super().__init__()

        self.args = args
        self.num_classes = 10

        if args.resize:
            self.transform = Compose([
                Resize([args.image_size, args.image_size]),
                ToTensor(),
                Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        else:
            self.transform = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        self.trainset = CIFAR10(root=args.dataset_root,
                                train=True,
                                download=True,
                                transform=self.transform)
        self.testset = CIFAR10(root=args.dataset_root,
                               train=False,
                               download=True,
                               transform=self.transform)

        self.trainset.num_classes = self.num_classes
        self.testset.num_classes = self.num_classes