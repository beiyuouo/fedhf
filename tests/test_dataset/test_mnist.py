#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    :   tests\test_dataset\test_mnist.py
# @Time    :   2022-05-03 12:01:30
# @Author  :   Bingjie Yan
# @Email   :   bj.yan.pa@qq.com
# @License :   Apache License 2.0


import numpy as np

from fedhf import Config
from fedhf.dataset import build_dataset


class TestMNIST(object):
    args = Config(num_classes=10, dataset="mnist")

    def test_mnist(self):
        print(self.args)
        dataset = build_dataset(self.args.dataset)(self.args)

        print(self.args.resize)

        assert dataset.num_classes == 10
        assert dataset.trainset.num_classes == 10
        assert dataset.testset.num_classes == 10

        assert len(dataset.trainset) == 60000
        assert len(dataset.testset) == 10000

        assert np.array(dataset.trainset[0][0]).shape == (1, 28, 28)
        assert np.array([dataset.trainset[0][1]]).shape == (1,)
        assert np.array(dataset.testset[0][0]).shape == (1, 28, 28)
        assert np.array([dataset.testset[0][1]]).shape == (1,)
