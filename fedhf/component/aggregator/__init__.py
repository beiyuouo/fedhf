#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    :   fedhf\component\aggregator\__init__.py
# @Time    :   2022-05-03 15:59:40
# @Author  :   Bingjie Yan
# @Email   :   bj.yan.pa@qq.com
# @License :   Apache License 2.0

__all__ = [
    "build_aggregator",
    "SyncAggregator",
    "AsyncAggregator",
    "aggregator_factory",
    "BaseAggregator",
]

from .base_aggregator import BaseAggregator
from .sync_aggregator import SyncAggregator
from .async_aggregator import AsyncAggregator

aggregator_factory = {
    "base": BaseAggregator,
    "sync": SyncAggregator,
    "async": AsyncAggregator,
}


def build_aggregator(agg_name: str):
    if agg_name not in aggregator_factory.keys():
        raise ValueError(f"Unknown aggregator name: {agg_name}")
    agg = aggregator_factory[agg_name]
    return agg
