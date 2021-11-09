#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   fedhf\component\client\simulated_client.py
@Time    :   2021-10-26 21:45:21
@Author  :   Bingjie Yan
@Email   :   bj.yan.pa@qq.com
@License :   Apache License 2.0
"""


from .base_client import BaseClient
from fedhf.component.evaluator import Evaluator
from fedhf.component.trainer import Trainer
from fedhf.model import build_loss, build_model, build_optimizer


class SimulatedClient(BaseClient):
    def __init__(self, args) -> None:
        self.args = args

        self.trainer = Trainer(args)
        self.evaluator = Evaluator(args)

    def train(self, data, model):
        self.trainer.train(data, model)
