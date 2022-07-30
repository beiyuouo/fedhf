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
    args = Config()

    def test_fedavg(self):
        pass
