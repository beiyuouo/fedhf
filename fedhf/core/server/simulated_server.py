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

from torch.utils.data.dataloader import DataLoader
from fedhf.component.aggregator import build_aggregator
from fedhf.component.selector import build_selector
from fedhf.component.evaluator import Evaluator
from fedhf.component.serializer import Serializer, Deserializer
from fedhf.model import build_criterion, build_model, build_optimizer
from fedhf.component.logger import Logger

from .base_server import BaseServer


class SimulatedServer(BaseServer):
    def __init__(self, args) -> None:
        self.args = args

        self.selector = build_selector(self.args.selector)(self.args)
        self.aggregator = build_aggregator(self.args.agg)(self.args)

        self.model = build_model(self.args.model)(self.args)
        self.evaluator = Evaluator(self.args)
        self.logger = Logger(self.args)

    def select(self, client_list: list):
        return self.selector.select(client_list)

    def update(self, model):
        result = self.aggregator.agg(
            Serializer.serialize_model(self.model),
            Serializer.serialize_model(model),
            server_model_version=self.model.get_model_version(),
            client_model_version=model.get_model_version())

        # print(self.model.get_model_version(), model.get_model_version())
        Deserializer.deserialize_model(self.model, result['param'])
        self.model.set_model_version(result['model_version'])
        self.model.set_model_time(result['model_time'])
        # print(result['model_version'], result['model_time'])
        self.logger.info(
            f'get model version{result["model_version"]} at time {result["model_time"]}'
        )
        return deepcopy(self.model)

    def evaluate(self, dataset):
        dataloader = DataLoader(dataset,
                                batch_size=self.args.batch_size,
                                shuffle=False)
        return self.evaluator.evaluate(dataloader=dataloader,
                                       model=self.model,
                                       device=self.args.device)