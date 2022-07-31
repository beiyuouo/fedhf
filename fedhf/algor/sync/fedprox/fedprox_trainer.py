#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    :   fedhf\algor\sync\fedprox\fedprox_trainer.py
# @Time    :   2022-07-17 01:37:44
# @Author  :   Bingjie Yan
# @Email   :   bj.yan.pa@qq.com
# @License :   Apache License 2.0


import time
from copy import deepcopy

from fedhf.component.trainer import DefaultTrainer


class FedProxTrainer(DefaultTrainer):
    def __init__(self, args) -> None:
        super(FedProxTrainer, self).__init__(args)

        self.rho = self.args[self.args.algor].get("rho") or 0.005
        self.args[self.args.algor].update({"rho": self.rho})

    def train_epoch(self, dataloader):
        losses = []
        self.logger.info(
            f"client:{self.client_id} training on epoch {self.epoch+1}/{self.args.num_epochs} loss: {0 if len(self.train_loss)==0 else self.train_loss[-1]:.5f}"
        )
        for idx, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(self.args.device)
            labels = labels.to(self.args.device)
            outputs = self.model(inputs)

            l2_reg = self._calc_l2_reg(self.model_, self.model)
            loss = self.criterion(outputs, labels) + l2_reg * self.rho / 2.0
            self.optimizer.zero_grad()

            if self.encryptor is not None:
                self.encryptor.clip_grad(self.model)

            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())

        self.train_loss.append(sum(losses) / len(losses))

    def _calc_l2_reg(self, global_model, model):
        l2_reg = 0
        for p1, p2 in zip(global_model.parameters(), model.parameters()):
            l2_reg += (p1 - p2).norm(2)
        return l2_reg
