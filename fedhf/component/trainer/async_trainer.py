#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   fedhf\component\trainer\trainer.py
@Time    :   2021-10-26 21:42:28
@Author  :   Bingjie Yan
@Email   :   bj.yan.pa@qq.com
@License :   Apache License 2.0
"""

import wandb

from copy import deepcopy

from tqdm import tqdm
from fedhf.component.logger import Logger
from fedhf.model import build_criterion, build_optimizer
from .base_train import BaseTrainer


class AsyncTrainer(BaseTrainer):
    def __init__(self, args) -> None:
        super().__init__()
        self.args = args
        self.optim = build_optimizer(self.args.optim)
        self.crit = build_criterion(self.args.train_loss)
        self.logger = Logger(self.args)

    def set_device(self, gpus, device):
        if len(gpus) > 1:
            pass
        else:
            pass

    def train(self, dataloader, model, num_epochs, client_id=None, gpus=[], device='cpu'):
        if len(gpus) > 1:
            pass
        else:
            pass

        model_ = deepcopy(model)
        model = model.to(device)
        model_ = model_.to(device)
        optim = self.optim(params=model.parameters(),
                           lr=self.args.lr,
                           momentum=self.args.momentum)
        crit = self.crit()

        self.logger.info(f'Start training on {client_id}')

        train_loss = []
        pbar = tqdm(total=num_epochs * len(dataloader))
        model.train()
        for epoch in range(num_epochs):
            losses = []
            pbar.set_description(f'Client:{client_id} Training on Epoch {epoch+1}/{num_epochs}')
            for idx, (inputs, labels) in enumerate(dataloader):
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)

                # self.logger.info(f'labels: {labels}, outputs: {outputs}')

                l2_reg = self._calc_l2_reg(model_, model)

                loss = crit(outputs, labels) + l2_reg * self.args.fedasync_rho / 2

                # self.logger.info(
                #    f'ce: {crit(outputs, labels)}, l2_reg: {l2_reg}, loss: {loss.item()}')

                optim.zero_grad()
                loss.backward()
                optim.step()

                losses.append(loss.item())

                pbar.update(1)

            train_loss.append(sum(losses) / len(dataloader.dataset))
            # self.logger.info(
            #    f'Client:{client_id} Epoch:{epoch+1}/{num_epochs} Loss:{train_loss[-1]}'
            #)

        self.logger.info(f'Client:{client_id} Train Loss:{train_loss}')
        if self.args.use_wandb:
            data = [[x, y] for (x, y) in zip(range(1, num_epochs + 1), train_loss)]
            table = wandb.Table(data=data, columns=["epoch", "train_loss"])
            self.logger.to_wandb({
                f"train at client {client_id} model_version {model.get_model_version()}":
                wandb.plot.line(table,
                                "epoch",
                                "train_loss",
                                title=f"train loss at client {client_id}")
            })

        return {'train_loss': train_loss, 'model': model}

    def _calc_l2_reg(self, global_model, model):
        l2_reg = 0
        for p1, p2 in zip(global_model.parameters(), model.parameters()):
            l2_reg += (p1 - p2).norm(2)
        return l2_reg