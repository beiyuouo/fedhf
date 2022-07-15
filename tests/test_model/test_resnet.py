#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    :   tests\test_model\test_resnet.py
# @Time    :   2022-05-03 12:15:10
# @Author  :   Bingjie Yan
# @Email   :   bj.yan.pa@qq.com
# @License :   Apache License 2.0

import torch
from torch.utils.data import DataLoader

from fedhf import Config
from fedhf.model import build_model, build_optimizer
from fedhf.dataset import build_dataset


class TestResnet(object):
    args = Config(
        model="resnet_mnist", num_classes=10, dataset="mnist", gpus="-1", resize=True, input_c=1, image_size=224
    )

    def test_resnet(self):
        model = build_model(self.args.model)(self.args)
        print(model)

        assert model.__class__.__name__ == "ResNetMNIST"
        assert model.net.__class__.__name__ == "ResNet"
        assert model.num_classes == 10
        assert model.net.fc.out_features == 10

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
