#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    :   tests\test_component\test_aggregator.py
# @Time    :   2022-07-15 16:11:10
# @Author  :   Bingjie Yan
# @Email   :   bj.yan.pa@qq.com
# @License :   Apache License 2.0

import pytest
import os
import torch
from fedhf import model
from fedhf import Config, Serializer
import fedhf
from fedhf.component import build_aggregator
from fedhf.model import build_model


@pytest.mark.order(3)
class TestAggregator(object):
    def test_agg_fedavg(self):
        args = fedhf.init(algor="fedavg")
