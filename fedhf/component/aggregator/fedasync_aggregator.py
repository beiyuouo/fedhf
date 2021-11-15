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

from fedhf import model

from .base_aggregator import BaseAggregator


class FedAsyncAggregator(BaseAggregator):
    def __init__(self, args) -> None:
        super().__init__()
        self.args = args
        self.stragegy = args.fedasync_strategy
        self.a = None
        self.b = None
        self.alpha = args.fedasync_alpha if args.fedasync_alpha else 0.5

    def agg(self, server_param, client_param, **kwargs):
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
        time_delta = kwargs["server_model_version"] - kwargs[
            "client_model_version"]
        if self.args.fedasync_strategy == "constant":
            return torch.mul(self.alpha, 1)
        elif self.strategy == "hinge" and self.b is not None and self.a is not None:
            if time_delta <= self.b:
                return torch.mul(self.alpha, 1)
            else:
                return torch.mul(self.alpha,
                                 1 / (self.a * ((time_delta - self.b) + 1)))
        elif self.strategy == "polynomial" and self.a is not None:
            return (time_delta + 1)**(-self.a)
        else:
            raise ValueError("Unknown strategy: {}".format(self.args.strategy))

    def _check_agg(self):
        return True
