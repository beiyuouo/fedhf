#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    :   fedhf\component\trainer\trainer.py
# @Time    :   2022-05-03 16:01:26
# @Author  :   Bingjie Yan
# @Email   :   bj.yan.pa@qq.com
# @License :   Apache License 2.0

import time
from copy import deepcopy

from torch.utils.data import DataLoader
from fedhf.model import build_criterion, build_optimizer
from .base_trainer import BaseTrainer


class DefaultTrainer(BaseTrainer):
    def __init__(self, args) -> None:
        super(DefaultTrainer, self).__init__(args)

    def train_epoch(self, dataloader):
        losses = []
        self.logger.info(
            f"client:{self.client_id} training on epoch {self.epoch+1}/{self.args.num_epochs} loss: {0 if len(self.train_loss)==0 else self.train_loss[-1]:.5f}"
        )
        for idx, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(self.args.device)
            labels = labels.to(self.args.device)

            outputs = self.model(inputs)

            loss = self.criterion(outputs, labels)
            self.optimizer.zero_grad()

            if self.encryptor is not None:
                self.encryptor.clip_grad(self.model)

            loss.backward()
            self.optimizer.step()

            losses.append(loss.item())

        self.train_loss.append(sum(losses) / len(losses))

    def train(
        self,
        dataloader,
        model,
        num_epochs,
        client_id=None,
        gpus=[],
        device="cpu",
        encryptor=None,
        **kwargs,
    ):
        self.args.num_epochs = num_epochs
        self.client_id = client_id
        self.args.device = device
        self.encryptor = encryptor

        model_ = deepcopy(model)
        self.model = model.to(device)
        self.model_ = model_.to(device)
        if self.args.optim == "sgd":
            self.optimizer = self.optim(
                params=model.parameters(),
                lr=self.args.lr,
                momentum=self.args.momentum,
                weight_decay=self.args.weight_decay,
            )
        else:
            self.optimizer = self.optim(params=model.parameters(), lr=self.args.lr)
        self.criterion = self.crit()

        if self.lr_scheduler:
            self.scheduler = self.lr_scheduler(self.optimizer, self.args.lr_step)
        else:
            self.scheduler = None

        self.logger.info(f"start training on {client_id}")

        self.train_loss = []
        model.train()
        for epoch in range(num_epochs):
            self.epoch = epoch
            self.train_epoch(dataloader)

            if self.lr_scheduler:
                self.scheduler.step()

        self.logger.info(f"client:{self.client_id} train loss:{self.train_loss}")
        return {"train_loss": self.train_loss, "model": model}
