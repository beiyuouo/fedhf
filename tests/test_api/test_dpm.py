#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    :   tests\test_api\test_dpm.py
# @Time    :   2022-05-02 23:36:03
# @Author  :   Bingjie Yan
# @Email   :   bj.yan.pa@qq.com
# @License :   Apache License 2.0

from copy import copy, deepcopy
import numpy as np
import torch
import torch.nn as nn

from fedhf import Config, dpm

from fedhf.model.nn import MLP
from fedhf.dataset.random import RandomDataset


class TestDPM:
    args = Config()

    def test_calculate_sensitivity(self):
        lr = 0.1
        clip = 10
        data_size = 100
        sensitivity = dpm.calculate_sensitivity(lr, clip, data_size)
        assert sensitivity == 2 * lr * clip / data_size

    def test_none(self):
        dpm.build_mechanism("none", dpm.calculate_sensitivity(0.1, 10, 100), 100, 0.1)
        assert np.all(
            dpm.build_mechanism(
                "none", dpm.calculate_sensitivity(0.1, 10, 100), 100, 0.1
            )
            == 0
        )

    def test_gaussian_noise(self):
        dpm.build_mechanism(
            "gaussian", dpm.calculate_sensitivity(0.1, 10, 100), 100, 0.1, delta=0.1
        )

    def test_laplace_noise(self):
        dpm.build_mechanism(
            "laplace", dpm.calculate_sensitivity(0.1, 10, 100), 100, 0.1
        )

    def test_none_clip(self):
        model = MLP(None, input_dim=10 * 10, output_dim=10)
        data = RandomDataset(None, 100, (1, 10, 10), 10)

        model.train()
        optim = torch.optim.SGD(model.parameters(), lr=1)
        crit = nn.CrossEntropyLoss()

        for epoch in range(1):
            loss = torch.tensor(0)
            for i, (x, y) in enumerate(data):
                x = x.view(1, 1, 10, 10)
                y = y.view(-1)
                output = model(x)
                loss = loss + crit(output, y)

            loss.backward()
            optim.step()
            # optim.zero_grad()

        grads = {k: v.grad.detach().numpy().copy() for k, v in model.named_parameters()}
        dpm.build_clip_grad("none", model, 0.1)

        # check grad is not changed

        for k, v in model.named_parameters():
            assert np.allclose(grads[k], v.grad.detach().numpy())

    def test_gaussian_clip(self):
        model = MLP(None, input_dim=10 * 10, output_dim=10)
        data = RandomDataset(None, 100, (1, 10, 10), 10)

        model.train()
        optim = torch.optim.SGD(model.parameters(), lr=1)
        crit = nn.CrossEntropyLoss()

        for epoch in range(1):
            loss = torch.tensor(0)
            for i, (x, y) in enumerate(data):
                x = x.view(1, 1, 10, 10)
                y = y.view(-1)
                output = model(x)
                loss = loss + crit(output, y)
            optim.zero_grad()
            loss.backward()
            optim.step()

        clip = 1e-8
        grads = {k: v.grad.detach().numpy().copy() for k, v in model.named_parameters()}
        print(grads)
        dpm.build_clip_grad("gaussian", model, clip)

        for k, v in model.named_parameters():
            assert np.all(np.abs(v.grad.detach().numpy()) <= clip)
            assert np.any(v.grad.detach().numpy() != grads[k])

    def test_laplace_clip(self):
        model = MLP(None, input_dim=10 * 10, output_dim=10)
        data = RandomDataset(None, 100, (1, 10, 10), 10)

        model.train()
        optim = torch.optim.SGD(model.parameters(), lr=1)
        crit = nn.CrossEntropyLoss()

        for epoch in range(1):
            loss = torch.tensor(0)
            for i, (x, y) in enumerate(data):
                x = x.view(1, 1, 10, 10)
                y = y.view(-1)
                output = model(x)
                loss = loss + crit(output, y)
            optim.zero_grad()
            loss.backward()
            optim.step()

        clip = 1e-8
        grads = {k: v.grad.detach().numpy().copy() for k, v in model.named_parameters()}
        print(grads)
        dpm.build_clip_grad("laplace", model, clip)

        for k, v in model.named_parameters():
            assert np.all(np.abs(v.grad.detach().numpy()) <= clip)
            assert np.any(v.grad.detach().numpy() != grads[k])
