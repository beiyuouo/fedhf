#!/usr/bin/env python
# -*- encoding: utf-8 -*-
""" 
@File    :   tests\test_core\test_server.py 
@Time    :   2021-11-15 19:21:17 
@Author  :   Bingjie Yan 
@Email   :   bj.yan.pa@qq.com 
@License :   Apache License 2.0 
"""

from fedhf.api import opts
from fedhf.core import SimulatedServer
from fedhf.dataset import build_dataset, ClientDataset
from fedhf.model import build_model


class TestServer(object):
    args = opts().parse([
        '--num_clients', '10', '--select_ratio', '0.5', '--num_classes', '10',
        '--model', 'mlp', '--dataset', 'mnist', '--batch_size', '1', '--optim',
        'sgd', '--lr', '0.01', '--loss', 'ce', '--gpus', '-1', '--test'
    ])

    def test_simulated_server(self):
        self.args.model = 'mlp'
        server = SimulatedServer(self.args)

        selected_clients = server.select(
            client_list=[i for i in range(self.args.num_clients)])

        assert len(selected_clients) > 0

        model = build_model(self.args.model)(self.args)
        model.set_model_version(0)

        assert server.model.get_model_version() == 0
        assert model.get_model_version() == 0

        model = server.update(model)

        assert server.model.get_model_version() == 1

        dataset = build_dataset(self.args.dataset)(self.args)
        dataset_small = ClientDataset(dataset.testset,
                                      [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        result = server.evaluate(dataset=dataset_small)
        assert 'test_loss' in result.keys()
