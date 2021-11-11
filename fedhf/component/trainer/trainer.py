#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   fedhf\component\trainer\trainer.py
@Time    :   2021-10-26 21:42:28
@Author  :   Bingjie Yan
@Email   :   bj.yan.pa@qq.com
@License :   Apache License 2.0
"""

from tqdm import tqdm
import wandb
from torch.utils.data import DataLoader
from fedhf.component.logger import Logger
from fedhf.model import build_criterion, build_optimizer
from .base_train import BaseTrainer


class Trainer(BaseTrainer):
    def __init__(self, args) -> None:
        super().__init__()
        self.args = args
        self.optim = build_optimizer(self.args.optim)
        self.crit = build_criterion(self.args.loss)
        self.logger = Logger(self.args)

    def set_device(self, gpus, device):
        if len(gpus) > 1:
            pass
        else:
            pass

    def train(self,
              dataloader,
              model,
              num_epochs,
              client_id=None,
              gpus=[],
              device='cpu'):
        if len(gpus) > 1:
            pass
        else:
            pass

        model = model.to(device)
        optim = self.optim(params=model.parameters(), lr=self.args.lr)
        crit = self.crit()

        train_loss = []
        for epoch in range(num_epochs):
            model.train()
            losses = []
            for inputs, labels in tqdm(
                    dataloader,
                    desc=f'Client:{client_id} Epoch:{epoch+1}/{num_epochs}'):
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = crit(outputs, labels)

                optim.zero_grad()
                loss.backward()
                optim.step()

                losses.append(loss.item())

            train_loss.append(sum(losses) / len(losses))

        return {'train_loss': train_loss, 'model': model}
