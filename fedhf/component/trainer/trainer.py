#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   fedhf\component\trainer\trainer.py
@Time    :   2021-10-26 21:42:28
@Author  :   Bingjie Yan
@Email   :   bj.yan.pa@qq.com
@License :   Apache License 2.0
"""

import tqdm
import wandb
from torch.utils.data import DataLoader
from fedhf.component.logger import Logger
from .base_train import BaseTrainer


class Trainer(BaseTrainer):
    def __init__(self) -> None:
        pass

    @staticmethod
    def train(dataloader,
              model,
              optim,
              crit,
              num_epochs,
              client_id=None,
              device='cpu'):
        model = model.to(device)
        optim = optim.to(device)
        crit = crit.to(device)

        train_loss = []
        for epoch in range(num_epochs):
            model.train()
            losses = []
            for inputs, labels in tqdm(dataloader,
                                       desc=f'Epoch {epoch+1}/{num_epochs}'):
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
