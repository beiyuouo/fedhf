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
        '--model', 'unet', '--model_pretrained', '--dataset', 'cifar10', '--gpus', '-1', '--task',
        'classification', '--resize', '--input_c', '3', '--output_c', '1', '--image_size', '224'
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
        torch.save(model.state_dict(), './model.pth')

        _model_state_dict = torch.load('./model.pth')
        # print(torch.load('./model.pth'))
        # print('-' * 10)
        # print(model.state_dict())
        # print(model.load())

        for k, v in model.state_dict().items():
            assert v.shape == _model_state_dict[k].shape
            assert v.dtype == _model_state_dict[k].dtype
            assert v.view(-1).eq(_model_state_dict[k].view(-1)).all()
            # assert v.device == _model_state_dict[k].device

        model.load()

        for k, v in model.state_dict().items():
            assert v.shape == _model_state_dict[k].shape
            assert v.dtype == _model_state_dict[k].dtype
            assert v.view(-1).eq(_model_state_dict[k].view(-1)).all()
            # assert v.device == _model_state_dict[k].device

        # assert torch.load('./model.pth') == model.state_dict()
        # assert torch.load(self.args.model_path)['state_dict'] == model.state_dict()