#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    :   fedhf\component\aggregator\async_aggregator.py
# @Time    :   2022-05-03 15:59:45
# @Author  :   Bingjie Yan
# @Email   :   bj.yan.pa@qq.com
# @License :   Apache License 2.0

import time

import torch
import torch.nn as nn

from fedhf import model

from fedhf.api import Logger
from .base_aggregator import BaseAggregator


class AsyncAggregator(BaseAggregator):

    def __init__(self, args) -> None:
        super(AsyncAggregator, self).__init__(args)
        self.alpha = 0.6

    def agg(self, server_param: torch.Tensor, client_param: torch.Tensor, **kwargs):
        if not self._check_agg():
            return

        alpha = self.alpha
        new_param = torch.mul(1 - alpha, server_param) + \
                                torch.mul(alpha, client_param)

        result = {
            'param':
                new_param,
            'model_version':
                kwargs["server_model_version"] +
                1 if "server_model_version" in kwargs.keys() else 0,
            'model_time':
                time.time()
        }
        return result

    def _check_agg(self):
        return True
