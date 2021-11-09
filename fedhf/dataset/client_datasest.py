#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   fedhf\dataset\base_datasest.py
@Time    :   2021-11-09 00:14:38
@Author  :   Bingjie Yan
@Email   :   bj.yan.pa@qq.com
@License :   Apache License 2.0
"""

from torch.utils.data import Dataset


class ClientDataset(Dataset):
    def __init__(self, dataset, data_dict) -> None:
        super().__init__()
        self.dataset = dataset
        self.data_dict = data_dict

        for attr in self.dataset.__dict__:
            if attr not in self.__dict__ and not attr.startswith('__') and not attr in ['trainset', 'testset']:
                self.__dict__[attr] = getattr(self.dataset, attr)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[self.data_dict[index]]
