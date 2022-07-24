#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    :   fedhf\component\aggregator\sync_aggregator.py
# @Time    :   2022-05-03 16:00:07
# @Author  :   Bingjie Yan
# @Email   :   bj.yan.pa@qq.com
# @License :   Apache License 2.0

import time
import numpy as np
import torch
import torch.nn as nn

from fedhf import Config
from .base_aggregator import BaseAggregator


class SyncAggregator(BaseAggregator):
    def __init__(self, args) -> None:
        super(SyncAggregator, self).__init__(args)

        self.num_clients_per_round = args.num_clients_per_round
        self._model_cached = []
        self._model_counter = 0
        self._model_weight = [1.0 for i in range(self.num_clients_per_round)]

    def agg(self, server_param: torch.Tensor, client_param: torch.Tensor, **kwargs):
        self._model_cached.append(client_param)
        self._model_counter += 1
        kwargs = Config(**kwargs)

        if not self._check_agg():
            return

        self._model_weight = np.array(self._model_weight, dtype=np.float32)
        self._model_weight = self._model_weight / self._model_weight.sum()  # normalize

        self.logger.info("aggregate models")

        new_param = torch.zeros_like(server_param)
        for i in range(self.num_clients_per_round):
            new_param += self._model_weight[i] * self._model_cached[i]

        self._model_cached = []
        self._model_counter = 0
        self._model_weight = [1.0 for i in range(self.num_clients_per_round)]

        result = {
            "param": new_param,
        }
        return result

    def _check_agg(self) -> bool:
        return self._model_counter == self.num_clients_per_round
