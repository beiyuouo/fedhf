#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   fedhf\component\evaluator\evaluator.py
@Time    :   2021-10-26 20:47:11
@Author  :   Bingjie Yan
@Email   :   bj.yan.pa@qq.com
@License :   Apache License 2.0
"""

import torch
from tqdm import tqdm

from fedhf.model import build_criterion, build_optimizer
from fedhf.component.logger import Logger
from .base_evaluator import BaseEvaluator


class Evaluator(BaseEvaluator):
    def __init__(self, args) -> None:
        self.args = args
        self.crit = build_criterion(self.args.loss)
        self.logger = Logger(self.args)

    def set_device(self, gpus, device):
        if len(gpus) > 1:
            pass
        else:
            pass

    def evaluate(self, dataloader, model, client_id=None, gpus=[], device='cpu'):
        if len(gpus) > 1:
            pass
        else:
            pass
        if not client_id:
            client_id = -1
        model = model.to(device)
        crit = self.crit()

        self.logger.info(f'Start evaluation on {client_id}')

        model.eval()
        losses = 0.0
        acc = 0.0
        for inputs, labels in tqdm(dataloader, desc=f'Test on client {client_id}'):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = crit(outputs, labels)

            _, predicted = torch.max(outputs, 1)
            acc += torch.sum(predicted == labels).item()

            losses += loss.item()

        losses /= len(dataloader)
        # self.logger.info(f'Client {client_id} test loss: {losses:.4f}, acc: {acc}')
        acc /= len(dataloader.dataset)

        self.logger.info(f'Evaluation on {client_id} finished')

        if self.args.use_wandb:
            if client_id == -1:
                self.logger.to_wandb({
                    'acc on server': acc,
                    'loss on server': losses,
                    'epoch': model.get_model_version()
                })

        return {'test_loss': losses, 'test_acc': acc}
