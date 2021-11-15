#!/usr/bin/env python
# -*- encoding: utf-8 -*-
""" 
@File    :   tests\test_component\test_evaluator.py 
@Time    :   2021-11-11 13:33:59 
@Author  :   Bingjie Yan 
@Email   :   bj.yan.pa@qq.com 
@License :   Apache License 2.0 
"""

import torch
from torch.utils.data import DataLoader

from fedhf.api import opts
from fedhf.component import Evaluator
from fedhf.dataset import build_dataset, ClientDataset
from fedhf.model import build_model


class TestEvaluator:
    args = opts().parse([
        '--num_classes', '10', '--model', 'mlp', '--dataset', 'mnist',
        '--batch_size', '1', '--optim', 'sgd', '--lr', '0.01', '--loss', 'ce',
        '--gpus', '-1'
    ])

    def test_evaluate(self):
        dataset = build_dataset(self.args.dataset)(self.args, resize=False)

        client_id = 0

        model = build_model(self.args.model)(self.args)

        client_dataset = ClientDataset(dataset.trainset,
                                       [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        dataloader = DataLoader(client_dataset,
                                batch_size=self.args.batch_size)

        evaluator = Evaluator(self.args)
        evaluator.evaluate(dataloader=dataloader,
                           model=model,
                           device=self.args.device)

    def test_evaluator_on_gpu(self):
        if not torch.cuda.is_available():
            return

        self.args.gpus = '0'
        self.args.device = torch.device('cuda:0')
        dataset = build_dataset(self.args.dataset)(self.args, resize=False)

        client_id = 0

        model = build_model(self.args.model)(self.args)

        client_dataset = ClientDataset(dataset.trainset,
                                       [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        dataloader = DataLoader(client_dataset,
                                batch_size=self.args.batch_size)

        evaluator = Evaluator(self.args)
        evaluator.evaluate(dataloader=dataloader,
                           model=model,
                           device=self.args.device)