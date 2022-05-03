#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    :   tests\test_core\test_client.py
# @Time    :   2022-05-03 15:58:11
# @Author  :   Bingjie Yan
# @Email   :   bj.yan.pa@qq.com
# @License :   Apache License 2.0

from torch.utils.data import DataLoader

from fedhf.api import opts
from fedhf.core import SimulatedClient
from fedhf.model import build_model
from fedhf.dataset import build_dataset, ClientDataset


class TestClient(object):
    args = opts().parse([
        '--num_classes', '10', '--model', 'mlp', '--dataset', 'mnist', '--num_local_epochs', '1',
        '--batch_size', '1', '--optim', 'sgd', '--lr', '0.01', '--loss', 'ce', '--gpus', '-1'
    ])

    def test_simulated_client(self):
        self.args.model = 'mlp'

        client_id = 0

        dataset = build_dataset(self.args.dataset)(self.args)
        client = SimulatedClient(self.args, client_id=client_id, data_size=len(dataset.trainset))

        model = build_model(self.args.model)(self.args)

        client_dataset = ClientDataset(dataset.trainset, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

        model = client.train(data=client_dataset, model=model, device=self.args.device)

        assert model is not None

        client.evaluate(data=client_dataset, model=model, device=self.args.device)
