#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    :   tests\test_model\test_mlp.py
# @Time    :   2022-05-03 12:15:04
# @Author  :   Bingjie Yan
# @Email   :   bj.yan.pa@qq.com
# @License :   Apache License 2.0

import torch
from torch.utils.data import DataLoader

import fedhf
from fedhf import Config
from fedhf.model import build_model, build_optimizer
from fedhf.dataset import build_dataset


class TestMLP(object):
    args = fedhf.init(model="mlp", num_classes=10, dataset="mnist", gpus="-1")

    def test_mlp(self):
        model = build_model(self.args.model)(self.args)
        print(model)

        assert model.__class__.__name__ == "MLP"
        assert model.layer_hidden.out_features == 10

        dataset = build_dataset(self.args.dataset)(self.args)
        dataloader = DataLoader(dataset.trainset, batch_size=1, shuffle=False)

        model = model.to(self.args.device)
        model.train()

        for data, target in dataloader:
            output = model(data)
            assert output.shape == (1, 10)
            assert output.dtype == torch.float32
            assert output.device == torch.device("cpu")
            break

        model.save()
