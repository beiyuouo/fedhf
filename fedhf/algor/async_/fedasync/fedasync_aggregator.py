#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    :   fedhf\algor\async_\fedasync\fedasync_aggregator.py
# @Time    :   2022-07-15 13:17:23
# @Author  :   Bingjie Yan
# @Email   :   bj.yan.pa@qq.com
# @License :   Apache License 2.0


import time
import torch
import torch.nn as nn
from ....component.aggregator.async_aggregator import AsyncAggregator


class FedAsyncAggregator(AsyncAggregator):
    def __init__(self, args) -> None:
        super(FedAsyncAggregator, self).__init__(args)

        self.algor = args.algor
        assert self.algor != "" and self.algor is not None

        self.stragegy = args[self.algor].strategy
        assert self.stragegy != "" and self.stragegy in ["constant", "hinge", "polynomial"]

        self.a = args[self.algor].a if args[self.algor].a is not None else None
        self.b = args[self.algor].b if args[self.algor].b is not None else None
        self.alpha = args[self.algor].alpha if args[self.algor].alpha is not None else None

        # asserts
        assert self.alpha is not None, "alpha is required"
        if self.stragegy == "hinge":
            assert self.a is not None, "a is required for strategy hinge"
            assert self.b is not None, "b is required for strategy hinge"
        elif self.stragegy == "polynomial":
            assert self.a is not None, "a is required for strategy polynomial"

    def agg(self, server_param: torch.Tensor, client_param: torch.Tensor, **kwargs):
        if not self._check_agg():
            return

        # model version is required for staleness
        if "server_model_version" not in kwargs.keys():
            raise ValueError("Missing key: server_model_version")
        if "client_model_version" not in kwargs.keys():
            raise ValueError("Missing key: client_model_version")

        server_model_version = kwargs["server_model_version"]
        client_model_version = kwargs["client_model_version"]

        if server_model_version < client_model_version:
            raise ValueError(
                f"server_model_version {server_model_version} < client_model_version {client_model_version}"
            )

        alpha = self._get_alpha(staleness=server_model_version - client_model_version)
        new_param = torch.mul(1 - alpha, server_param) + torch.mul(alpha, client_param)

        # assert torch.equal(new_param, server_param) == False
        # assert torch.equal(new_param, client_param) == False

        self.logger.info(
            f"Aggregated server model version: {server_model_version}, client model version: {client_model_version}"
        )
        self.logger.info(f"FedAsyncAggregator agg alpha: {alpha} with stragegy: {self.stragegy}")

        result = {
            "param": new_param,
            "model_version": server_model_version + 1,
            "model_time": time.time(),
        }
        return result

    def _get_alpha(self, staleness: int):
        if self.stragegy == "constant":
            return torch.mul(self.alpha, 1)
        elif self.stragegy == "hinge" and self.b is not None and self.a is not None:
            if staleness <= self.b:
                return torch.mul(self.alpha, 1)
            else:
                return torch.mul(self.alpha, 1 / (self.a * ((staleness - self.b) + 1)))
        elif self.stragegy == "polynomial" and self.a is not None:
            return torch.mul(self.alpha, (staleness + 1) ** (-self.a))
        else:
            raise ValueError("Unknown strategy: {}".format(self.stragegy))
