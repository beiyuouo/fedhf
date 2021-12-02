#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   fedhf\component\aggregator\aggregator.py
@Time    :   2021-10-26 20:36:08
@Author  :   Bingjie Yan
@Email   :   bj.yan.pa@qq.com
@License :   Apache License 2.0
"""
import time
import torch
import torch.nn as nn

from .sync_aggregator import SyncAggregator


class FedAvgAggregator(SyncAggregator):
    def __init__(self, args) -> None:
        super(FedAvgAggregator, self).__init__(args)

    def agg(self, server_param: torch.Tensor, client_param: torch.Tensor, **kwargs):
        self._model_cached.append(client_param)
        self._model_counter += 1

        if "weight" not in kwargs.keys():
            kwargs["weight"] = 1 / self.num_clients_per_round

        self._model_weight[self._model_counter - 1] = kwargs["weight"]

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
