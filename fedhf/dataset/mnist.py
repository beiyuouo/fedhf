#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   fedhf\dataset\mnist.py
@Time    :   2021-11-08 22:41:04
@Author  :   Bingjie Yan
@Email   :   bj.yan.pa@qq.com
@License :   Apache License 2.0
"""

from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize


class MNISTDataset(object):
    def __init__(self, args) -> None:
        super().__init__()

        self.args = args
        self.num_classes = 10
        self.transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])

        self.trainset = MNIST(root=args.dataset_root, train=True,
                              download=True, transform=self.transform)
        self.testset = MNIST(root=args.dataset_root, train=False,
                             download=True, transform=self.transform)
