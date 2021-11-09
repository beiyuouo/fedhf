#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   fedhf\component\aggregator\async_aggregator.py
@Time    :   2021-10-28 11:56:57
@Author  :   Bingjie Yan
@Email   :   bj.yan.pa@qq.com
@License :   Apache License 2.0
"""

from .base_aggregator import BaseAggregator


class AsyncAggregator(BaseAggregator):
    def __init__(self) -> None:
        super().__init__()

    def agg(self, model, grads):
        pass

    def check_agg(self):
        pass
