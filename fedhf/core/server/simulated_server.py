#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    :   fedhf\core\server\simulated_server.py
# @Time    :   2022-05-03 03:34:11
# @Author  :   Bingjie Yan
# @Email   :   bj.yan.pa@qq.com
# @License :   Apache License 2.0

import time
from copy import deepcopy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from fedhf.api import Serializer, Deserializer

from .base_server import BaseServer


class SimulatedServer(BaseServer):
    def __init__(self, args) -> None:
        super(SimulatedServer, self).__init__(args)

    def update(self, model: nn.Module, **kwargs):
        # self.logger.info(f'Update model with {kwargs}')

        if self.model.get_model_version() == 0:
            self.model = deepcopy(model)

        result = self.aggregator.agg(
            Serializer.serialize_model(self.model),
            Serializer.serialize_model(model),
            **kwargs,
        )

        if not result:
            self.logger.info("it's not time to update.")
            return
        # print(self.model.get_model_version(), model.get_model_version())
        Deserializer.deserialize_model(self.model, result["param"])

        # self.model = self.encryptor.encrypt_model(self.model)

        self.model.set_model_version(self.model.get_model_version() + 1)
        self.model.set_model_time(time.time())
        # print(result['model_version'], result['model_time'])
        self.logger.info(
            f"get model version {self.model.get_model_version()} at time {self.model.get_model_time()}"
        )
        return

    def train(self, data: Dataset, model: nn.Module, device="cpu", **kwargs):
        dataloader = DataLoader(data, batch_size=self.args.batch_size)

        result = self.evaluator.evaluate(
            dataloader=dataloader, model=model, client_id=self.client_id, device=device
        )
        if "model" in result:
            model = result["model"]
            result.pop("model")

        self.logger.info(
            f"finish evaluating on client {self.client_id}, result: {result}"
        )
        return result
