#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    :   tests\test_algor\test_fedasync.py
# @Time    :   2022-07-20 15:05:38
# @Author  :   Bingjie Yan
# @Email   :   bj.yan.pa@qq.com
# @License :   Apache License 2.0


import pytest
import fedhf

from fedhf import Config


@pytest.mark.order(-1)
class TestFedAsync:
    args = Config()

    def test_fedasync(self):
        pass
