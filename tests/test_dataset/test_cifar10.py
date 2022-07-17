#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    :   tests\test_dataset\test_cifar10.py
# @Time    :   2022-05-03 12:01:37
# @Author  :   Bingjie Yan
# @Email   :   bj.yan.pa@qq.com
# @License :   Apache License 2.0

import numpy as np

import fedhf
from fedhf import Config
from fedhf.dataset import build_dataset


class TestCIFAR10(object):
    args = fedhf.init(num_classes=10, dataset="cifar10")

    def test_cifar10(self):
        dataset = build_dataset(self.args.dataset)(self.args)

        print(self.args.resize)

        assert dataset.num_classes == 10
        assert dataset.trainset.num_classes == 10
        assert dataset.testset.num_classes == 10

        assert len(dataset.trainset) == 50000
        assert len(dataset.testset) == 10000

        assert np.array(dataset.trainset[0][0]).shape == (3, 32, 32)
        assert np.array([dataset.trainset[0][1]]).shape == (1,)
        assert np.array(dataset.testset[0][0]).shape == (3, 32, 32)
        assert np.array([dataset.testset[0][1]]).shape == (1,)
