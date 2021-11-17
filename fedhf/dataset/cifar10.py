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
    def __init__(self, args, resize=None) -> None:
        super().__init__()

        self.args = args
        self.num_classes = 10
        resize = args.resize if resize is None else resize

        if resize:
            self.transform = Compose([
                Resize(args.image_size),
                ToTensor(),
                Normalize((0.49139968, 0.48215841, 0.44653091),
                          (0.24703223, 0.24348513, 0.26158784))
            ])
        else:
            self.transform = Compose([
                ToTensor(),
                Normalize((0.49139968, 0.48215841, 0.44653091),
                          (0.24703223, 0.24348513, 0.26158784))
            ])

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