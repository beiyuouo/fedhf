#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    :   fedhf\component\sampler\noniid_sampler.py
# @Time    :   2022-08-14 15:04:02
# @Author  :   Bingjie Yan
# @Email   :   bj.yan.pa@qq.com
# @License :   Apache License 2.0


import numpy as np

from fedhf.dataset import ClientDataset
from .base_sampler import BaseSampler


class NonIIDSampler(BaseSampler):
    default_args = {"noniid_sampler": {"alpha": 0.1, "least_samples": 100}}

    def __init__(self, args) -> None:
        super(NonIIDSampler, self).__init__(args)

    def sample(self, train_dataset, test_dataset=None, export=True, split=True):
        self.add_default_args()

        if split:
            train_idxs, test_idxs = self.split_dataset(range(len(train_dataset)))
            test_dataset = train_dataset

        client_data_train = {i: [] for i in range(-1, self.args.num_clients)}
        client_data_test = {i: [] for i in range(-1, self.args.num_clients)}

        dataset_label = train_dataset.train_labels
        # dataset label in train_idxs
        dataset_label = [dataset_label[i] for i in train_idxs]

        min_size = 0
        K = self.args.num_classes
        N = len(dataset_label)
        least_samples = self.args.noniid_sampler["least_samples"]
        alpha = self.args.noniid_sampler["alpha"]
        num_clients = self.args.num_clients

        # Code Reference: https://github.com/TsingZ0/PFL-Non-IID/blob/master/dataset/utils/dataset_utils.py
        # License: MIT

        while min_size < least_samples:
            idx_batch = [[] for _ in range(num_clients)]
            for k in range(K):
                idx_k = np.where(dataset_label == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
                proportions = np.array(
                    [
                        p * (len(idx_j) < N / num_clients)
                        for p, idx_j in zip(proportions, idx_batch)
                    ]
                )
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [
                    idx_j + idx.tolist()
                    for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))
                ]
                min_size = min([len(idx_j) for idx_j in idx_batch])

        for j in range(num_clients):
            client_data_train[j] = idx_batch[j]

        client_data_test[-1] = test_idxs

        train_data, test_data = {
            i: ClientDataset(train_dataset, list(client_data_train[i]))
            for i in client_data_train.keys()
        }, {
            i: ClientDataset(test_dataset, list(client_data_test[i]))
            for i in client_data_test.keys()
        }

        if export:
            self.export_data_partition(client_data_train, client_data_test)

        return train_data, test_data
