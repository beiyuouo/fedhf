#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    :   tests\test_model\test_cnn_cifar10.py
# @Time    :   2022-05-03 12:14:53
# @Author  :   Bingjie Yan
# @Email   :   bj.yan.pa@qq.com
# @License :   Apache License 2.0

import torch
from torch.utils.data import DataLoader

import fedhf
from fedhf import Config
from fedhf.model import build_model, build_optimizer
from fedhf.dataset import build_dataset


class TestCNNCIFAR10(object):
    args = fedhf.init(
        model="cnn2_cifar10",
        num_classes=10,
        dataset="cifar10",
        model_pretrained=True,
        gpus="-1",
        resize=True,
        image_size=32,
        input_c=3,
    )

    def test_cnn2cifar10(self):
        model = build_model(self.args.model)(self.args)
        print(model)

        assert model.__class__.__name__ == "CNN2CIFAR10"
        assert model.num_classes == 10

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
