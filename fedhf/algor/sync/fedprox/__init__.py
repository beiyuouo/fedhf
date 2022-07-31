#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    :   fedhf\algor\sync\fedprox\__init__.py
# @Time    :   2022-07-15 13:17:47
# @Author  :   Bingjie Yan
# @Email   :   bj.yan.pa@qq.com
# @License :   Apache License 2.0

import os
from fedhf import Config
from .fedprox_aggregator import FedProxAggregator
from .fedprox_trainer import FedProxTrainer

components = {
    "agg": {"fedprox": FedProxAggregator},
    "trainer": {"fedprox": FedProxTrainer},
}

default_args = Config()
default_args.load(os.path.join(os.path.dirname(__file__), "default_args.yaml"))
