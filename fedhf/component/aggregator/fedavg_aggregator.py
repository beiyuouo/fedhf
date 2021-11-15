#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   fedhf\component\aggregator\aggregator.py
@Time    :   2021-10-26 20:36:08
@Author  :   Bingjie Yan
@Email   :   bj.yan.pa@qq.com
@License :   Apache License 2.0
"""

from .base_aggregator import BaseAggregator


class FedAvgAggregator(BaseAggregator):
    def __init__(self, args) -> None:
        self.args = args

    def agg(self, model, grads):
        pass

    def _check_agg(self):
        pass
