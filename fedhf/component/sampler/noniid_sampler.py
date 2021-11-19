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

        idxs = np.arange(self.args.num_clients * num_items)
        labels = np.array(dataset.targets)
        idx_label = np.vstack((idxs, labels))
        idx_label = idx_label[:, idx_label[1, :].argsort()]
        idxs = idx_label[0, :]
        labels = idx_label[1, :]

        client_data_dict, all_idxs = {}, [i for i in range(len(dataset))]
        for i in range(self.args.num_clients):
            client_data_dict[i] = idxs[i * num_items:(i + 1) * num_items]
            all_idxs = list(set(all_idxs) - set(client_data_dict[i]))

        return [
            ClientDataset(dataset, client_data_dict[i])
            for i in range(self.args.num_clients)
        ]
