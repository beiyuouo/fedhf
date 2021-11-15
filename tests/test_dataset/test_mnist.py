#!/usr/bin/env python
# -*- encoding: utf-8 -*-
""" 
@File    :   tests\test_dataset\test_mnist.py
@Time    :   2021-11-10 17:10:20
@Author  :   Bingjie Yan
@Email   :   bj.yan.pa@qq.com
@License :   Apache License 2.0
"""

import numpy as np

from fedhf.dataset import build_dataset
from fedhf.api import opts


class TestMNIST(object):
    args = opts().parse([
        '--num_classes', '10', '--dataset_root', './dataset',
        '--dataset_download', 'True', '--dataset', 'mnist', '--resize', False
    ])

    def test_mnist(self):
        dataset = build_dataset(self.args.dataset)(self.args)

        print(self.args.resize)

        assert dataset.num_classes == 10
        assert dataset.trainset.num_classes == 10
        assert dataset.testset.num_classes == 10

        assert len(dataset.trainset) == 60000
        assert len(dataset.testset) == 10000

        assert np.array(dataset.trainset[0][0]).shape == (1, 28, 28)
        assert np.array([dataset.trainset[0][1]]).shape == (1, )
        assert np.array(dataset.testset[0][0]).shape == (1, 28, 28)
        assert np.array([dataset.testset[0][1]]).shape == (1, )