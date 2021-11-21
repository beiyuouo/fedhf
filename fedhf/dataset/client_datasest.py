#!/usr/bin/env python
# -*- encoding: utf-8 -*-
""" 
@File    :   fedhf\dataset\client_datasest.py 
@Time    :   2021-11-10 23:00:32 
@Author  :   Bingjie Yan 
@Email   :   bj.yan.pa@qq.com 
@License :   Apache License 2.0 
"""

from typing import List
from torch.utils.data import Dataset


class ClientDataset(Dataset):
    def __init__(self, dataset, data_dict: List) -> None:
        super().__init__()
        self.dataset = dataset
        self.data_dict = data_dict

        for attr in self.dataset.__dict__:
            if attr.startswith('__') or attr in self.__dict__:
                continue
            # print(attr)
            setattr(self, attr, getattr(self.dataset, attr))

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, index):
        return self.dataset[self.data_dict[index]]
