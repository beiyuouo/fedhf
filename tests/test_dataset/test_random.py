#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    :   tests\test_dataset\test_random.py
# @Time    :   2022-05-03 12:01:24
# @Author  :   Bingjie Yan
# @Email   :   bj.yan.pa@qq.com
# @License :   Apache License 2.0

import numpy as np
import torch

import fedhf
from fedhf import Config
from fedhf.dataset import build_dataset


class TestRandom(object):
    args = fedhf.init(num_classes=10, dataset="random")

    def test_random(self):
        print(self.args)
        length = 10
        data_shape = (10,)
        dataset = build_dataset(self.args.dataset)(self.args, length, data_shape)

        assert dataset.length == length
        assert dataset.data_shape == data_shape
        assert dataset.num_classes == 10

        assert len(dataset) == length
        assert dataset[0][0].shape == data_shape
        assert 0 <= dataset[0][1].item() < dataset.num_classes
        assert dataset[0][1].shape == torch.Size([])
