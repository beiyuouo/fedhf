#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    :   fedhf\algor\async_\fedasync\__init__.py
# @Time    :   2022-07-15 12:54:17
# @Author  :   Bingjie Yan
# @Email   :   bj.yan.pa@qq.com
# @License :   Apache License 2.0

import os
from fedhf.api import EmptyConfig
from .fedasync_aggregator import FedAsyncAggregator
from .fedasync_trainer import FedAsyncTrainer

components = {
    "agg": {"fedasync": FedAsyncAggregator},
    "trainer": {"fedasync": FedAsyncTrainer},
}

default_params = EmptyConfig()
default_params.load(os.path.join(os.path.dirname(__file__), "default_params.yaml"))
