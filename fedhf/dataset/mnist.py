#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    :   fedhf\dataset\mnist.py
# @Time    :   2022-05-03 16:06:35
# @Author  :   Bingjie Yan
# @Email   :   bj.yan.pa@qq.com
# @License :   Apache License 2.0

from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
import torch


class MNISTDataset(object):
    def __init__(self, args) -> None:
        super().__init__()

        self.args = args
        self.num_classes = 10

        if args.resize:
            self.transform = Compose([Resize([args.image_size, args.image_size]), ToTensor()])
        else:
            self.transform = Compose([ToTensor()])

        self.trainset = MNIST(root=args.data_dir, train=True, download=True, transform=self.transform)
        self.testset = MNIST(root=args.data_dir, train=False, download=True, transform=self.transform)

        self.trainset.num_classes = self.num_classes
        self.testset.num_classes = self.num_classes

    def get_train_norm(self) -> float:
        """Never used"""
        means = []
        stds = []
        for img, label in self.trainset:
            means.append(torch.mean(img))
            stds.append(torch.std(img))

        mean = torch.mean(torch.tensor(means))
        std = torch.mean(torch.tensor(stds))
        return {"mean": mean, "std": std}
