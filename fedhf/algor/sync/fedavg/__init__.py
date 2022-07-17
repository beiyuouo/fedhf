#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    :   fedhf\algor\sync\fedavg\__init__.py
# @Time    :   2022-07-15 13:17:37
# @Author  :   Bingjie Yan
# @Email   :   bj.yan.pa@qq.com
# @License :   Apache License 2.0

import os
from fedhf import Config
from .fedavg_aggregator import FedAvgAggregator

components = {"agg": {"fedavg": FedAvgAggregator}}

default_params = Config()
default_params.load(os.path.join(os.path.dirname(__file__), "default_params.yaml"))
