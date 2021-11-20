#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   tests\test_component\test_sampler.py
@Time    :   2021-11-09 20:03:30
@Author  :   Bingjie Yan
@Email   :   bj.yan.pa@qq.com
@License :   Apache License 2.0
"""

from fedhf.component import build_sampler
from fedhf.api import opts
from fedhf.dataset import build_dataset


class TestSampler(object):
    args = opts().parse([
        '--num_clients', '100', '--dataset', 'mnist', '--num_classes', '10', '--dataset_root',
        './dataset', '--sampler_num_classes', '2', '--sampler_num_samples', '300'
    ])

    def test_random_sampler(self):
        sampler = build_sampler('random')(self.args)

        assert sampler is not None
        assert sampler.__class__.__name__ == 'RandomSampler'

        dataset = build_dataset(self.args.dataset)(self.args)
        train_data = sampler.sample(dataset.trainset)

        assert len(train_data) == self.args.num_clients
        assert len(train_data[0]) == len(dataset.trainset) // self.args.num_clients

    def test_noniid_sampler(self):
        sampler = build_sampler('non-iid')(self.args)

        assert sampler is not None
        assert sampler.__class__.__name__ == 'NonIIDSampler'

        dataset = build_dataset(self.args.dataset)(self.args)
        train_data = sampler.sample(dataset.trainset)

        print(train_data[0])

        assert len(train_data) == self.args.num_clients
        assert len(train_data[0]) == len(dataset.trainset) // self.args.num_clients
