#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   fedhf\component\evaluator\evaluator.py
@Time    :   2021-10-26 20:47:11
@Author  :   Bingjie Yan
@Email   :   bj.yan.pa@qq.com
@License :   Apache License 2.0
"""

import tqdm

from .base_evaluator import BaseEvaluator


class Evaluator(BaseEvaluator):
    def __init__(self, args) -> None:
        self.args = args

    def evaluate(dataloader, model, optim, crit, client_id=None, device='cpu'):
        model = model.to(device)
        optim = optim.to(device)
        crit = crit.to(device)

        model.eval()
        losses = 0.0
        for inputs, labels in tqdm(dataloader
                                    , desc=f'Test on client{client_id}'):
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
