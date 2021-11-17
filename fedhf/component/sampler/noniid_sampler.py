#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   fedhf\component\sampler\noniid_sampler.py
@Time    :   2021-11-08 21:10:21
@Author  :   Bingjie Yan
@Email   :   bj.yan.pa@qq.com
@License :   Apache License 2.0
"""

import numpy as np

from fedhf.dataset import ClientDataset
from .base_sampler import BaseSampler


class NonIIDSampler(BaseSampler):
    def __init__(self, args) -> None:
        self.args = args

    def sample(self, dataset):
        num_items = int(len(dataset) / self.args.num_clients)
        client_data_dict = {}
        for i in range(self.args.num_clients):
            client_data_dict[i] = [
                k
                for k in range(i *
                               num_items, min((i + 1) *
                                              num_items, len(dataset)))
            ]

        return [
            ClientDataset(dataset, client_data_dict[i])
            for i in range(self.args.num_clients)
        ]
