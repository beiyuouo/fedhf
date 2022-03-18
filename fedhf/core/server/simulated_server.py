#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   fedhf\component\server\simulated_server.py
@Time    :   2021-10-26 21:44:30
@Author  :   Bingjie Yan
@Email   :   bj.yan.pa@qq.com
@License :   Apache License 2.0
"""

from copy import deepcopy
import re
import torch
import torch.nn as nn

from fedhf.api import Logger, Serializer, Deserializer
from fedhf.component import build_aggregator, build_selector, Evaluator
from fedhf.model import build_criterion, build_model, build_optimizer

from .base_server import BaseServer


class SimulatedServer(BaseServer):

    def __init__(self, args) -> None:
        super(SimulatedServer, self).__init__(args)

    def update(self, model: nn.Module, **kwargs):
        if self.model.model_version == -1:
            self.model = deepcopy(model)

        result = self.aggregator.agg(Serializer.serialize_model(self.model),
                                     Serializer.serialize_model(model), **kwargs)

        if not result:
            self.logger.info('It\'s not time to update.')
            return
        # print(self.model.get_model_version(), model.get_model_version())
        Deserializer.deserialize_model(self.model, result['param'])
        self.model.set_model_version(result['model_version'])
        self.model.set_model_time(result['model_time'])
        # print(result['model_version'], result['model_time'])
        self.logger.info(
            f'get model version {result["model_version"]} at time {result["model_time"]}')
        return
