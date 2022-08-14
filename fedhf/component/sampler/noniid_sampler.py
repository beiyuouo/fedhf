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
    default_args = {
        "noniid_sampler": {"alpha": 0.5, "least_samples": 20, "split": False}
    }

    def __init__(self, args) -> None:
        super(NonIIDSampler, self).__init__(args)

    def sample(self, train_dataset, test_dataset=None, export=True, split=None):
        self.add_default_args()

        if split is True or self.args.noniid_sampler.get("split", True):
            train_idxs, test_idxs = self.split_dataset(list(range(len(train_dataset))))
            test_dataset = train_dataset
        else:
            train_idxs = list(range(len(train_dataset)))
            test_idxs = list(range(len(test_dataset)))

        # print("train_idxs:", train_idxs)
        # print("test_idxs:", test_idxs)

        client_data_train = {i: [] for i in range(-1, self.args.num_clients)}
        client_data_test = {i: [] for i in range(-1, self.args.num_clients)}

        dataset_label = train_dataset.targets.numpy()
        # dataset label in train_idxs
        dataset_label = np.array([dataset_label[i] for i in train_idxs])
        # print("dataset_label:", dataset_label)
        # print("len(dataset_label):", len(dataset_label))
        # print(
        #     "np.where(np.array(dataset_label) == 0)[0]:",
        #     np.where(np.array(dataset_label) == 0)[0],
        # )

        min_size = 0
        K = self.args.num_classes
        N = len(dataset_label)
        least_samples = self.args.noniid_sampler["least_samples"]
        alpha = self.args.noniid_sampler["alpha"]
        num_clients = self.args.num_clients

        # print(
        #     "K: {}, N: {}, least_samples: {}, alpha: {}".format(
        #         K, N, least_samples, alpha
        #     )
        # )

        # Code Reference: https://github.com/TsingZ0/PFL-Non-IID/blob/master/dataset/utils/dataset_utils.py
        # License: MIT

        while min_size < least_samples:
            idx_batch = [[] for _ in range(num_clients)]
            for k in range(K):
                idx_k = np.where(dataset_label == k)[0]
                # print(f"k: {k}, idx_k: {idx_k}")
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

            # print("min_size:", min_size)
            # print("idx_batch:", idx_batch)

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

        self.logger.info("client_data_train: {}".format(client_data_train))
        self.logger.info("client_data_test: {}".format(client_data_test))

        if export:
            self.export_data_partition(client_data_train, client_data_test)

        return train_data, test_data


if __name__ == "__main__":
    from fedhf.api import Config
    from fedhf.dataset import build_dataset

    args = Config(
        num_clients=100,
        num_classes=10,
        dataset="mnist",
        gpus="-1",
        resize=False,
        data_dir="./../../../dataset",
        save_dir="./../../../runs/exp",
    )
    sampler = NonIIDSampler(args=args)

    assert sampler is not None
    assert sampler.__class__.__name__ == "NonIIDSampler"

    dataset = build_dataset(args.dataset)(args)
    train_data, test_data = sampler.sample(dataset.trainset, dataset.testset)

    assert len(train_data) == args.num_clients
    assert len(train_data[0]) == len(dataset.trainset) // args.num_clients
    assert len(test_data[1]) % (len(dataset.testset) // args.num_classes) == 0
