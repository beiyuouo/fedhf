#!/usr/bin/env python
# -*- encoding: utf-8 -*-
""" 
@File    :   fedhf\component\aggregator\sync_aggregator.py 
@Time    :   2021-12-02 12:56:12 
@Author  :   Bingjie Yan 
@Email   :   bj.yan.pa@qq.com 
@License :   Apache License 2.0 
"""

import time
import torch
import torch.nn as nn

from fedhf.component.logger import Logger

from .base_aggregator import BaseAggregator


class SyncAggregator(BaseAggregator):
    def __init__(self, args) -> None:
        super(SyncAggregator, self).__init__(args)

        self.num_clients_per_round = int(args.num_clients * args.select_ratio)
        self._model_cached = []
        self._model_counter = 0
        self._model_weight = [
            1 / self.num_clients_per_round for i in range(self.num_clients_per_round)
        ]

    def agg(self, server_param: torch.Tensor, client_param: torch.Tensor, **kwargs):
        self._model_cached.append(client_param)
        self._model_counter += 1

        if not self._check_agg():
            return

        self.logger.info('Aggregate models')

        new_param = torch.zeros_like(server_param)
        for i in range(self.num_clients_per_round):
            new_param += self._model_weight[i] * self._model_cached[i]

        self._model_cached = []
        self._model_counter = 0
        self._model_weight = [
            1 / self.num_clients_per_round for i in range(self.num_clients_per_round)
        ]

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

    def _check_agg(self) -> bool:
        return self._model_counter == self.num_clients_per_round
