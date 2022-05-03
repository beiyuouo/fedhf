#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   fedhf\dataset\__init__.py
@Time    :   2021-10-26 11:11:16
@Author  :   Bingjie Yan
@Email   :   bj.yan.pa@qq.com
@License :   Apache License 2.0
"""

from .mnist import MNISTDataset
from .cifar10 import CIFAR10Dataset
from .random import RandomDataset
from .client_datasest import ClientDataset

dataset_factory = {
    'mnist': MNISTDataset,
    'cifar10': CIFAR10Dataset,
    'random': RandomDataset,
}


def build_dataset(dataset_name: str):
    if dataset_name not in dataset_factory.keys():
        raise NotImplementedError

    dataset = dataset_factory[dataset_name]

    return dataset


__all__ = [
    'build_dataset', 'dataset_factory', 'MNISTDataset', 'CIFAR10Dataset', 'RandomDataset',
    'ClientDataset'
]
