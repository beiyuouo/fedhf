#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   fedhf\component\aggregator\async_aggregator.py
@Time    :   2021-10-28 11:56:57
@Author  :   Bingjie Yan
@Email   :   bj.yan.pa@qq.com
@License :   Apache License 2.0
"""

import torch

from .base_aggregator import BaseAggregator


class FedAsyncAggregator(BaseAggregator):
    def __init__(self, args) -> None:
        super().__init__()
        self.args = args
        self.model_time = 0

    def agg(self, server_param, client_param):
        if not self.check_agg():
            return
        
        alpha = self._get_alpha()
        new_param = torch.mul(1 - alpha, server_param) + \
                                torch.mul(alpha, client_param)
        self.model_time += 1

        return new_param
        
    def _get_alpha(self):
        if self.args.strategy == "constant":
            return torch.mul(self.alpha, 1)
        else:
            raise ValueError("Unknown strategy: {}".format(self.args.strategy))

    def check_agg(self):
        return True
