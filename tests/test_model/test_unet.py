#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    :   tests\test_model\test_unet.py
# @Time    :   2022-02-28 20:51:10
# @Author  :   Bingjie Yan
# @Email   :   bj.yan.pa@qq.com
# @License :   Apache License 2.0

import torch
from torch import optim
from torch.utils.data import DataLoader

from fedhf.api import opts
from fedhf.model import build_model, build_optimizer
from fedhf.dataset import build_dataset


class TestDenseNet(object):
    args = opts().parse([
        '--model', 'unet', '--model_pretrained', '--dataset', 'cifar10', '--gpus', '-1',
        '--task', 'classification', '--resize', '--input_c', '3', '--output_c', '1',
        '--image_size', '224'
    ])

    def test_desenet(self):
        model = build_model(self.args.model)(self.args)
        print(model)

        assert model.__class__.__name__ == 'UNet'
        assert model.input_c == 3
        assert model.output_c == 1

        dataset = build_dataset(self.args.dataset)(self.args)
        dataloader = DataLoader(dataset.trainset, batch_size=1, shuffle=False)

        model = model.to(self.args.device)
        model.train()
        for data, target in dataloader:
            output = model(data)
            assert output.shape == (1, 1, 224, 224)
            assert output.dtype == torch.float32
            assert output.device == torch.device('cpu')
            break

        model.save()