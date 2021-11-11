#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   fedhf\component\evaluator\evaluator.py
@Time    :   2021-10-26 20:47:11
@Author  :   Bingjie Yan
@Email   :   bj.yan.pa@qq.com
@License :   Apache License 2.0
"""

from tqdm import tqdm

from .base_evaluator import BaseEvaluator
from fedhf.model import build_criterion, build_optimizer
from fedhf.component.logger import Logger


class Evaluator(BaseEvaluator):
    def __init__(self, args) -> None:
        self.args = args
        self.optim = build_optimizer(self.args.optim)
        self.crit = build_criterion(self.args.loss)
        self.logger = Logger(self.args)

    def set_device(self, gpus, device):
        if len(gpus) > 1:
            pass
        else:
            pass

    def evaluate(self,
                 dataloader,
                 model,
                 client_id=None,
                 gpus=[],
                 device='cpu'):
        if len(gpus) > 1:
            pass
        else:
            pass
        if not client_id:
            client_id = -1
        model = model.to(device)
        optim = self.optim(params=model.parameters(), lr=self.args.lr)
        crit = self.crit()

        model.eval()
        losses = 0.0
        for inputs, labels in tqdm(dataloader,
                                   desc=f'Test on client {client_id}'):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = crit(outputs, labels)

            optim.zero_grad()
            loss.backward()
            optim.step()

            losses += loss.item()

        losses /= len(dataloader)

        return {'test_loss': losses}
