#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    :   tests\test_component\test_sampler.py
# @Time    :   2022-07-15 16:11:23
# @Author  :   Bingjie Yan
# @Email   :   bj.yan.pa@qq.com
# @License :   Apache License 2.0


from fedhf import Config
import fedhf
from fedhf.component import build_sampler
from fedhf.dataset import build_dataset


class TestSampler(object):

    args = fedhf.init(
        num_clients=100,
        dataset="mnist",
        num_classes=10,
        data_dir="./dataset",
        sampler_num_classes=2,
        sampler_num_samples=300,
    )

    def test_random_sampler(self):
        sampler = build_sampler("random")(self.args)

        assert sampler is not None
        assert sampler.__class__.__name__ == "RandomSampler"

        dataset = build_dataset(self.args.dataset)(self.args)
        train_data, test_data = sampler.sample(dataset.trainset, dataset.testset)

        assert len(train_data) == self.args.num_clients
        assert len(train_data[0]) == len(dataset.trainset) // self.args.num_clients
        assert len(test_data) == self.args.num_clients

    def test_noniid_sampler(self):
        sampler = build_sampler("non-iid")(self.args)

        assert sampler is not None
        assert sampler.__class__.__name__ == "NonIIDSampler"

        dataset = build_dataset(self.args.dataset)(self.args)
        train_data, test_data = sampler.sample(dataset.trainset)

        assert len(train_data) == self.args.num_clients
        assert len(train_data[0]) == len(dataset.trainset) // self.args.num_clients
        assert len(test_data) == self.args.num_clients

    def test_noniid_sampler_with_test_dataset(self):
        sampler = build_sampler("non-iid")(self.args)

        assert sampler is not None
        assert sampler.__class__.__name__ == "NonIIDSampler"

        dataset = build_dataset(self.args.dataset)(self.args)
        train_data, test_data = sampler.sample(dataset.trainset, dataset.testset)

        assert len(train_data) == self.args.num_clients
        assert len(train_data[0]) == len(dataset.trainset) // self.args.num_clients
        assert len(test_data[1]) % (len(dataset.testset) // self.args.num_classes) == 0
