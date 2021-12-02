#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   fedhf\component\aggregator\async_aggregator.py
@Time    :   2021-10-28 11:56:57
@Author  :   Bingjie Yan
@Email   :   bj.yan.pa@qq.com
@License :   Apache License 2.0
"""
import time

import torch
import torch.nn as nn

from fedhf import model

from fedhf.component.logger import Logger
from .async_aggregator import AsyncAggregator


class FedAsyncAggregator(AsyncAggregator):
    def __init__(self, args) -> None:
        super(FedAsyncAggregator, self).__init__(args)

        self.stragegy = args.fedasync_strategy
        self.a = args.fedasync_a if args.fedasync_a else None
        self.b = args.fedasync_b if args.fedasync_b else None
        self.alpha = args.fedasync_alpha if args.fedasync_alpha else 0.5

    def agg(self, server_param: torch.Tensor, client_param: torch.Tensor, **kwargs):
        if not self._check_agg():
            return
        if not self.stragegy == "constant":
            if "server_model_version" not in kwargs.keys():
                raise ValueError("Missing key: server_model_version")
            if "client_model_version" not in kwargs.keys():
                raise ValueError("Missing key: client_model_version")

            if kwargs["server_model_version"] < kwargs["client_model_version"]:
                raise ValueError("server_model_version < client_model_version")

        alpha = self._get_alpha(**kwargs)
        new_param = torch.mul(1 - alpha, server_param) + \
                                torch.mul(alpha, client_param)

        # assert torch.equal(new_param, server_param) == False
        # assert torch.equal(new_param, client_param) == False

        self.logger.info(
            f"Aggregated server model version: {kwargs['server_model_version']}, client model version: {kwargs['client_model_version']}"
        )
        self.logger.info(
            f"FedAsyncAggregator agg: alpha: {alpha} using stragegy: {self.stragegy}")

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

    def _get_alpha(self, **kwargs):
        staleness = kwargs["server_model_version"] - kwargs["client_model_version"]
        if self.args.fedasync_strategy == "constant":
            return torch.mul(self.alpha, 1)
        elif self.args.fedasync_strategy == "hinge" and self.b is not None and self.a is not None:
            if staleness <= self.b:
                return torch.mul(self.alpha, 1)
            else:
                return torch.mul(self.alpha, 1 / (self.a * ((staleness - self.b) + 1)))
        elif self.args.fedasync_strategy == "polynomial" and self.a is not None:
            return torch.mul(self.alpha, (staleness + 1)**(-self.a))
        else:
            raise ValueError("Unknown strategy: {}".format(self.args.strategy))