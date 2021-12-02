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
import time
from copy import deepcopy

from tqdm import tqdm
from torch.utils.data import DataLoader
from fedhf.model import build_criterion, build_optimizer
from .base_trainer import BaseTrainer


class Trainer(BaseTrainer):
    def __init__(self, args) -> None:
        super(Trainer, self).__init__(args)

    def train(self, dataloader, model, num_epochs, client_id=None, gpus=[], device='cpu'):
        if len(gpus) > 1:
            pass
        else:
            pass

        model_ = deepcopy(model)
        model = model.to(device)
        model_ = model_.to(device)
        if self.args.optim == 'sgd':
            optim = self.optim(params=model.parameters(),
                               lr=self.args.lr,
                               momentum=self.args.momentum,
                               weight_decay=self.args.weight_decay)
        else:
            optim = self.optim(params=model.parameters(), lr=self.args.lr)
        crit = self.crit()

        self.logger.info(f'Start training on {client_id}')

        train_loss = []
        pbar = tqdm(total=num_epochs * len(dataloader))
        model.train()
        for epoch in range(num_epochs):
            losses = []
            pbar.set_description(
                f'Client:{client_id} Training on Epoch {epoch+1}/{num_epochs} Loss: {0 if len(train_loss)==0 else train_loss[-1]:.5f}'
            )
            for idx, (inputs, labels) in enumerate(dataloader):
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)

                loss = crit(outputs, labels)
                optim.zero_grad()
                loss.backward()
                optim.step()

                losses.append(loss.item())
                pbar.update(1)

            train_loss.append(sum(losses) / len(losses))

        time.sleep(0.3)  # wait for pbar to update
        self.logger.info(f'Client:{client_id} Train Loss:{train_loss}')

        if self.args.use_wandb and self.args.wandb_log_client:
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