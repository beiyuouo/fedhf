#!/usr/bin/env python
# -*- encoding: utf-8 -*-
""" 
@File    :   tests\test_model\test_resnet.py 
@Time    :   2021-11-11 10:58:42 
@Author  :   Bingjie Yan 
@Email   :   bj.yan.pa@qq.com 
@License :   Apache License 2.0 
"""

import torch
from torch import optim
from torch.utils.data import DataLoader

from fedhf.api import opts
from fedhf.model import build_model, build_optimizer
from fedhf.dataset import build_dataset


class TestResnet(object):
    args = opts().parse([
        '--model', 'resnet', '--num_classes', '10', '--model_pretrained',
        'True', '--dataset', 'mnist', '--device', '-1', '--task',
        'classification'
    ])

    def test_resnet(self):
        model = build_model(self.args.model)(self.args)
        print(model)

        assert model.cnn.__class__.__name__ == 'ResNet'
        assert model.num_classes == 10
        assert model.cnn.fc.out_features == 10

        dataset = build_dataset(self.args.dataset)(self.args, resize=True)
        dataloader = DataLoader(dataset.trainset, batch_size=1, shuffle=False)
        for data, target in dataloader:
            output = model(data)
            assert output.shape == (1, 10)
            assert output.dtype == torch.float32
            assert output.device == torch.device('cpu')
            break