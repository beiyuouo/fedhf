#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    :   tests\test_algor\test_fedprox.py
# @Time    :   2022-07-31 10:03:19
# @Author  :   Bingjie Yan
# @Email   :   bj.yan.pa@qq.com
# @License :   Apache License 2.0


import pytest
import fedhf

from fedhf import Config


@pytest.mark.order(-1)
class TestFedProx:
    args = fedhf.init(
        debug=True,
        algor="fedprox",
        num_clients=3,
        num_rounds=1,
        num_epochs=1,
        model="mlp",
        dataset="mnist",
        scheme="sync",
        agg="fedprox",
        trainer="fedprox",
    )

    def test_fedprox(self):
        fedhf.run(self.args)
