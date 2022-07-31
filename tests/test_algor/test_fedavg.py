#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    :   tests\test_algor\test_fedavg.py
# @Time    :   2022-07-20 15:05:33
# @Author  :   Bingjie Yan
# @Email   :   bj.yan.pa@qq.com
# @License :   Apache License 2.0


import pytest
import fedhf

from fedhf import Config


@pytest.mark.order(-1)
class TestFedAvg:
    args = fedhf.init(
        debug=True,
        algor="fedavg",
        num_clients=3,
        num_rounds=1,
        num_epochs=1,
        model="mlp",
        dataset="mnist",
        scheme="sync",
        agg="fedavg",
    )

    def test_fedavg(self):
        fedhf.run(self.args)
