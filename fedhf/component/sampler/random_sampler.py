#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   fedhf\component\sampler\random_sampler.py
@Time    :   2021-10-28 14:17:50
@Author  :   Bingjie Yan
@Email   :   bj.yan.pa@qq.com
@License :   Apache License 2.0
"""

import numpy as np

from fedhf.dataset import ClientDataset
from .base_sampler import BaseSampler


class RandomSampler(BaseSampler):
    def __init__(self, args) -> None:
        super(RandomSampler, self).__init__(args)

    def sample(self, dataset):
        num_items = int(len(dataset) / self.args.num_clients)
        client_data_dict, all_idxs = {}, [i for i in range(len(dataset))]
        for i in range(self.args.num_clients):
            client_data_dict[i] = list(np.random.choice(all_idxs, num_items, replace=False))
            all_idxs = list(set(all_idxs) - set(client_data_dict[i]))

        return [
            ClientDataset(dataset, client_data_dict[i]) for i in range(self.args.num_clients)
        ]
