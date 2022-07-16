#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    :   fedhf\core\server\base_server.py
# @Time    :   2022-05-03 15:45:08
# @Author  :   Bingjie Yan
# @Email   :   bj.yan.pa@qq.com
# @License :   Apache License 2.0

from abc import ABC
from torch.utils.data.dataloader import DataLoader

from fedhf.api import Logger
from fedhf.component import (
    build_aggregator,
    build_selector,
    build_evaluator,
    build_encryptor,
)
from fedhf.model import build_model


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

        self.model = build_model(self.args.model)(self.args, model_version=0)
        self.evaluator = build_evaluator(self.args.evaluator)(self.args)
        if self.args.get("encryptor"):
            self.encryptor = build_encryptor(self.args.encryptor)(self.args)

        self.logger = Logger(self.args)

    def select(self, client_list: list):
        return self.selector.select(client_list)

    def evaluate(self, dataset, **kwargs):
        dataloader = DataLoader(dataset, batch_size=self.args.batch_size, shuffle=False)
        return self.evaluator.evaluate(
            dataloader=dataloader, model=self.model, device=self.args.device, **kwargs
        )
