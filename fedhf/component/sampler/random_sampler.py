#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    :   fedhf\component\sampler\random_sampler.py
# @Time    :   2022-05-03 16:00:48
# @Author  :   Bingjie Yan
# @Email   :   bj.yan.pa@qq.com
# @License :   Apache License 2.0

import numpy as np

from fedhf.dataset import ClientDataset
from .base_sampler import BaseSampler


class RandomSampler(BaseSampler):
    def __init__(self, args) -> None:
        super(RandomSampler, self).__init__(args)

    def sample(self, train_dataset, test_dataset=None):
        num_items = int(len(train_dataset) / self.args.num_clients)
        client_data_dict, all_idxs = {}, [i for i in range(len(train_dataset))]
        for i in range(self.args.num_clients):
            client_data_dict[i] = list(
                np.random.choice(all_idxs, num_items, replace=False)
            )
            all_idxs = list(set(all_idxs) - set(client_data_dict[i]))

        if test_dataset is not None:
            num_items_test = int(len(test_dataset) / self.args.num_clients)
            client_data_dict_test, all_idxs_test = {}, [
                i for i in range(len(test_dataset))
            ]
            for i in range(self.args.num_clients):
                client_data_dict_test[i] = list(
                    np.random.choice(all_idxs_test, num_items_test, replace=False)
                )
                all_idxs_test = list(set(all_idxs_test) - set(client_data_dict_test[i]))

        train_data = {
            i: ClientDataset(train_dataset, client_data_dict[i])
            for i in range(self.args.num_clients)
        }
        if test_dataset is not None:
            test_data = {
                i: ClientDataset(test_dataset, client_data_dict_test[i])
                for i in range(self.args.num_clients)
            }
        else:
            test_data = {i: None for i in range(self.args.num_clients)}

        return train_data, test_data
