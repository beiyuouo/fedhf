#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   fedhf\dataset\__init__.py
@Time    :   2021-10-26 11:11:16
@Author  :   Bingjie Yan
@Email   :   bj.yan.pa@qq.com
@License :   Apache License 2.0
"""

__all__ = ["build_dataset", "MNISTDataset", "ClientDataset"]


from mnist import MNISTDataset
from client_datasest import ClientDataset


dataset_factory = {
    'mnist': MNISTDataset
}


def build_dataset(dataset_name: str):
    if dataset_name not in dataset_factory.keys():
        raise NotImplementedError

    dataset = dataset_factory[dataset_name]

    return dataset
