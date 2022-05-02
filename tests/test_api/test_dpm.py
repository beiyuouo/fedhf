#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    :   tests\test_api\test_dpm.py
# @Time    :   2022-05-02 23:36:03
# @Author  :   Bingjie Yan
# @Email   :   bj.yan.pa@qq.com
# @License :   Apache License 2.0

import numpy as np
import torch
import torch.nn as nn

from fedhf.api import opts, dpm

from fedhf.model.nn import MLP


class TestDPM:
    args = opts().parse()

    def test_calculate_sensitivity(self):
        lr = 0.1
        clip = 10
        data_size = 100
        sensitivity = dpm.calculate_sensitivity(lr, clip, data_size)
        assert sensitivity == 2 * lr * clip / data_size

    def test_none(self):
        dpm.build_mechanism('none', dpm.calculate_sensitivity(0.1, 10, 100), 100, 0.1)
        assert np.all(
            dpm.build_mechanism('none', dpm.calculate_sensitivity(0.1, 10, 100), 100, 0.1) == 0)

    def test_gaussian_noise(self):
        dpm.build_mechanism('gaussian',
                            dpm.calculate_sensitivity(0.1, 10, 100),
                            100,
                            0.1,
                            delta=0.1)

    def test_laplace_noise(self):
        dpm.build_mechanism('laplace', dpm.calculate_sensitivity(0.1, 10, 100), 100, 0.1)
