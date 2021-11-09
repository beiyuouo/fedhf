#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   fedhf\component\server\simulated_server.py
@Time    :   2021-10-26 21:44:30
@Author  :   Bingjie Yan
@Email   :   bj.yan.pa@qq.com
@License :   Apache License 2.0
"""


from fedhf.component.aggregator import build_aggregator
from fedhf.component.selector import build_selector
from fedhf.model import build_loss, build_model, build_optimizer

from .base_server import BaseServer

class SimulatedServer(BaseServer):
    def __init__(self, args) -> None:
        self.args = args

        self.selector = build_selector(self.args.selector)()
        self.aggregator = build_aggregator(self.args.agg)(self.args)

        self.model = build_model(self.args.model)()

    def select(self, client_list: list):
        return self.selector.select(client_list)

    def update(self, model):
        self.aggregator.aggregate(self.model, model)
