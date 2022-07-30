#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    :   tests\test_component\test_trainer.py
# @Time    :   2022-07-15 16:11:32
# @Author  :   Bingjie Yan
# @Email   :   bj.yan.pa@qq.com
# @License :   Apache License 2.0

import pytest
import torch
from torch.utils.data import DataLoader

from fedhf import Config
import fedhf
from fedhf.component import DefaultTrainer as Trainer
from fedhf.model import build_model, build_criterion, build_optimizer
from fedhf.dataset import build_dataset, ClientDataset


@pytest.mark.order(3)
class TestTrainer(object):
    args = fedhf.init(
        num_classes=10,
        model="mlp",
        dataset="mnist",
        num_epochs=1,
        batch_size=1,
        optim="sgd",
        lr=0.01,
        loss="ce",
        gpus="-1",
    )

    def test_trainer_mlp(self):
        self.args.model = "mlp"
        self.args.resize = False
        dataset = build_dataset(self.args.dataset)(self.args)

        client_id = 0

        model = build_model(self.args.model)(self.args)

        client_dataset = ClientDataset(dataset.trainset, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        dataloader = DataLoader(client_dataset, batch_size=self.args.batch_size)

        trainer = Trainer(self.args)
        result = trainer.train(
            dataloader=dataloader,
            model=model,
            num_epochs=self.args.num_epochs,
            client_id=client_id,
            device=self.args.device,
        )
        train_loss = result["train_loss"]
        model = result["model"]
        print(train_loss)

    def test_trainer_on_gpu_mlp(self):
        self.args.model = "mlp"
        self.args.resize = False
        if not torch.cuda.is_available():
            return

        self.args.gpus = "0"
        self.args.device = torch.device("cuda:0")

        dataset = build_dataset(self.args.dataset)(self.args)

        client_id = 0

        model = build_model(self.args.model)(self.args)

        client_dataset = ClientDataset(dataset.trainset, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        dataloader = DataLoader(client_dataset, batch_size=self.args.batch_size)

        trainer = Trainer(self.args)
        result = trainer.train(
            dataloader=dataloader,
            model=model,
            num_epochs=self.args.num_epochs,
            client_id=client_id,
            device=self.args.device,
        )
        train_loss = result["train_loss"]
        model = result["model"]
        print(train_loss)

    def test_trainer_resnet(self):
        self.args.model = "resnet_mnist"
        self.args.resize = True
        dataset = build_dataset(self.args.dataset)(self.args)

        client_id = 0

        model = build_model(self.args.model)(self.args)

        client_dataset = ClientDataset(dataset.trainset, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        dataloader = DataLoader(client_dataset, batch_size=self.args.batch_size)

        trainer = Trainer(self.args)
        result = trainer.train(
            dataloader=dataloader,
            model=model,
            num_epochs=self.args.num_epochs,
            client_id=client_id,
            device=self.args.device,
        )
        train_loss = result["train_loss"]
        model = result["model"]
        print(train_loss)

    def test_trainer_on_gpu_resnet(self):
        self.args.model = "resnet_mnist"
        if not torch.cuda.is_available():
            return

        self.args.gpus = "0"
        self.args.device = torch.device("cuda:0")
        self.args.resize = True

        dataset = build_dataset(self.args.dataset)(self.args)

        client_id = 0

        model = build_model(self.args.model)(self.args)

        client_dataset = ClientDataset(dataset.trainset, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        dataloader = DataLoader(client_dataset, batch_size=self.args.batch_size)

        trainer = Trainer(self.args)
        result = trainer.train(
            dataloader=dataloader,
            model=model,
            num_epochs=self.args.num_epochs,
            client_id=client_id,
            device=self.args.device,
        )
        train_loss = result["train_loss"]
        model = result["model"]
        print(train_loss)
