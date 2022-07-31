#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    :   fedhf\component\sampler\noniid_sampler.py
# @Time    :   2022-05-03 16:00:43
# @Author  :   Bingjie Yan
# @Email   :   bj.yan.pa@qq.com
# @License :   Apache License 2.0

import numpy as np

from fedhf.dataset import ClientDataset
from .base_sampler import BaseSampler


class NonIIDSampler(BaseSampler):
    def __init__(self, args) -> None:
        super(NonIIDSampler, self).__init__(args)

    def sample(self, train_dataset, test_dataset=None):
        """[summary]

        Reference: https://github.com/jeremy313/non-iid-dataset-for-personalized-federated-learning

        Args:
            dataset ([type]): [description]
            test_dataset ([type], optional): [description]. Defaults to None.

        Returns:
            [type]: [description]
        """
        num_items_train = int(len(train_dataset) / self.args.num_clients)
        sampler_num_samples = (
            self.args.sampler_num_samples
            if self.args.get("sampler_num_samples") is not None
            else 1
        )
        sampler_num_samples = int(sampler_num_samples)
        sampler_num_classes = (
            self.args.sampler_num_classes
            if self.args.get("sampler_num_classes") is not None
            else 1
        )
        sampler_num_classes = int(sampler_num_classes)
        num_shards_train = len(train_dataset) // sampler_num_samples

        if test_dataset is not None:
            num_items_test = len(test_dataset) // self.args.num_classes

        client_data_dict = {}

        idx_shard = [i for i in range(num_shards_train)]

        idxs = np.arange(self.args.num_clients * num_items_train, dtype=np.int32)
        labels = np.array(train_dataset.targets)

        if test_dataset is not None:
            idxs_test = np.arange(len(test_dataset), dtype=np.int32)
            labels_test = np.array(test_dataset.targets)

        idx_label = np.vstack((idxs, labels))
        idx_label = idx_label[:, idx_label[1, :].argsort()]
        idxs = idx_label[0, :]
        labels = idx_label[1, :]

        if test_dataset is not None:
            idx_label_test = np.vstack((idxs_test, labels_test))
            idxs_labels_test = idx_label_test[:, idx_label_test[1, :].argsort()]
            idxs_test = idxs_labels_test[0, :]
            labels_test = idxs_labels_test[1, :]

        client_data_train = {i: np.array([]) for i in range(self.args.num_clients)}
        client_data_test = {i: np.array([]) for i in range(self.args.num_clients)}

        unbalance_rate = (
            1.0
            if self.args.get("sampler_unbalance_rate") is None
            else self.args.sampler_unbalance_rate
        )
        unbalance_rate = float(unbalance_rate)

        for i in range(self.args.num_clients):
            rand_set = set(
                np.random.choice(idx_shard, sampler_num_classes, replace=False)
            )
            idx_shard = list(set(idx_shard) - rand_set)

            unbalance_flag = 0
            client_data_train[i] = np.array([], dtype=np.int32)
            client_label = np.array([])

            for rand_item in rand_set:
                if unbalance_flag == 0:
                    client_data_train[i] = np.concatenate(
                        (
                            client_data_train[i],
                            idxs[
                                sampler_num_samples
                                * rand_item : int(sampler_num_samples * (rand_item + 1))
                            ],
                        ),
                        axis=0,
                    )
                    client_label = np.concatenate(
                        (
                            client_label,
                            labels[
                                sampler_num_samples
                                * rand_item : int(sampler_num_samples * (rand_item + 1))
                            ],
                        ),
                        axis=0,
                    )
                else:
                    client_data_train[i] = np.concatenate(
                        (
                            client_data_train[i],
                            idxs[
                                sampler_num_samples
                                * rand_item : int(
                                    sampler_num_samples * (rand_item + unbalance_rate)
                                )
                            ],
                        ),
                        axis=0,
                    )
                    client_label = np.concatenate(
                        (
                            client_label,
                            labels[
                                sampler_num_samples
                                * rand_item : int(
                                    sampler_num_samples * (rand_item + unbalance_rate)
                                )
                            ],
                        ),
                        axis=0,
                    )
                unbalance_flag = 1
            client_label_set = set(client_label)

            if test_dataset is not None:
                for label in client_label_set:
                    client_data_test[i] = np.concatenate(
                        (
                            client_data_test[i],
                            idxs_test[
                                int(label)
                                * num_items_test : int(label + 1)
                                * num_items_test
                            ],
                        ),
                        axis=0,
                    )

        if test_dataset is None:
            return {
                i: ClientDataset(train_dataset, list(client_data_train[i]))
                for i in range(self.args.num_clients)
            }, {i: None for i in range(self.args.num_clients)}
        else:
            return {
                i: ClientDataset(train_dataset, list(client_data_train[i]))
                for i in range(self.args.num_clients)
            }, {
                i: ClientDataset(test_dataset, list(client_data_test[i]))
                for i in range(self.args.num_clients)
            }
