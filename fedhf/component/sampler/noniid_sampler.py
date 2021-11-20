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
        """[summary]

        Reference: https://github.com/jeremy313/non-iid-dataset-for-personalized-federated-learning

        Args:
            dataset ([type]): [description]

        Returns:
            [type]: [description]
        """
        num_items = int(len(dataset) / self.args.num_clients)
        num_shards = len(dataset) // self.args.sampler_num_samples
        client_data_dict = {}

        idx_shard = [i for i in range(num_shards)]

        idxs = np.arange(self.args.num_clients * num_items)
        labels = np.array(dataset.targets)
        idx_label = np.vstack((idxs, labels))
        idx_label = idx_label[:, idx_label[1, :].argsort()]
        idxs = idx_label[0, :]
        labels = idx_label[1, :]

        client_data_dict, all_idxs = {}, [i for i in range(len(dataset))]
        for i in range(self.args.num_clients):
            rand_set = set(
                np.random.choice(idx_shard, self.args.sampler_num_classes, replace=False))
            idx_shard = list(set(idx_shard) - rand_set)

            unbalance_flag = 0
            client_data_dict[i] = []

            for rand in rand_set:
                if unbalance_flag == 0:
                    client_data_dict[i] = np.concatenate(
                        (client_data_dict[i],
                         idxs[rand * self.args.sampler_num_samples:(rand + 1) *
                              self.args.sampler_num_samples]),
                        axis=0)
                else:
                    client_data_dict[i] = np.concatenate(
                        (client_data_dict[i], idxs[rand * self.args.sampler_num_samples:int(
                            (rand + self.args.sampler_unbalance_rate) *
                            self.args.sampler_num_samples)]),
                        axis=0)
                unbalance_flag = 1

        return [
            ClientDataset(dataset, list(client_data_dict[i]))
            for i in range(self.args.num_clients)
        ]
