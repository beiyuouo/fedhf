#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   fedhf\component\trainer\trainer.py
@Time    :   2021-10-26 21:42:28
@Author  :   Bingjie Yan
@Email   :   bj.yan.pa@qq.com
@License :   Apache License 2.0
"""


from fedhf.model import build_loss, build_model, build_optimizer

from .base_train import BaseTrainer


class Trainer(BaseTrainer):
    def __init__(self, args) -> None:
        self.args = args

        self.optim = build_optimizer(self.optim)(
            self.model.parameters(), self.args.lr)
        self.loss = build_loss(self.args.loss)

    def train(self, data, model):
        pass
