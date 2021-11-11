#!/usr/bin/env python
# -*- encoding: utf-8 -*-
""" 
@File    :   tests\test_component\test_trainer.py 
@Time    :   2021-11-11 12:29:11 
@Author  :   Bingjie Yan 
@Email   :   bj.yan.pa@qq.com 
@License :   Apache License 2.0 
"""

import torch
from torch.utils.data import DataLoader

from fedhf.api import opts
from fedhf.component.trainer import Trainer
from fedhf.model import build_model, build_criterion, build_optimizer
from fedhf.dataset import build_dataset, ClientDataset


class TestTrainer(object):
    args = opts().parse([
        '--num_classes', '10', '--model', 'mlp', '--dataset', 'mnist',
        '--num_local_epochs', '1', '--batch_size', '1', '--optim', 'sgd',
        '--lr', '0.01', '--loss', 'ce', '--gpus', '-1'
    ])

    def test_trainer(self):
        dataset = build_dataset(self.args.dataset)(self.args)

        client_id = 0

        model = build_model(self.args.model)(self.args)

        client_dataset = ClientDataset(dataset.trainset,
                                       [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        dataloader = DataLoader(client_dataset,
                                batch_size=self.args.batch_size)

        trainer = Trainer(self.args)
        result = trainer.train(dataloader=dataloader,
                               model=model,
                               num_epochs=self.args.num_local_epochs,
                               client_id=client_id,
                               device=self.args.device)
        train_loss = result['train_loss']
        model = result['model']
        print(train_loss)

    def test_trainer_on_gpu(self):
        if not torch.cuda.is_available():
            return

        self.args.gpus = '0'
        self.args.device = torch.device('cuda:0')

        dataset = build_dataset(self.args.dataset)(self.args)

        client_id = 0

        model = build_model(self.args.model)(self.args)

        client_dataset = ClientDataset(dataset.trainset,
                                       [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        dataloader = DataLoader(client_dataset,
                                batch_size=self.args.batch_size)

        trainer = Trainer(self.args)
        result = trainer.train(dataloader=dataloader,
                               model=model,
                               num_epochs=self.args.num_local_epochs,
                               client_id=client_id,
                               device=self.args.device)
        train_loss = result['train_loss']
        model = result['model']
        print(train_loss)