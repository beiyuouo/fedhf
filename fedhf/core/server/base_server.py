#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   fedhf\component\server\base_server.py
@Time    :   2021-10-26 11:07:00
@Author  :   Bingjie Yan
@Email   :   bj.yan.pa@qq.com
@License :   Apache License 2.0
"""

from abc import ABC
from torch.utils.data.dataloader import DataLoader

from fedhf.api import Logger, Serializer, Deserializer
from fedhf.component import build_aggregator, build_selector, Evaluator
from fedhf.model import build_criterion, build_model, build_optimizer


class AbsServer(ABC):
    def __init__(self) -> None:
        super().__init__()

    def update(self):
        raise NotImplementedError

    def select(self):
        raise NotImplementedError

    def evaluate(self):
        raise NotImplementedError


class BaseServer(AbsServer):
    def __init__(self, args) -> None:
        self.args = args

        self.selector = build_selector(self.args.selector)(self.args)
        self.aggregator = build_aggregator(self.args.agg)(self.args)

        self.model = build_model(self.args.model)(self.args)
        self.evaluator = Evaluator(self.args)
        self.logger = Logger(self.args)

    def select(self, client_list: list):
        return self.selector.select(client_list)

    def evaluate(self, dataset):
        dataloader = DataLoader(dataset, batch_size=self.args.batch_size, shuffle=False)
        return self.evaluator.evaluate(dataloader=dataloader,
                                       model=self.model,
                                       device=self.args.device)
