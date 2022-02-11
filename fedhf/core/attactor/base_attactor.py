#!/usr/bin/env python
# -*- encoding: utf-8 -*-
""" 
@File    :   fedhf\core\attactor\base_attactor.py 
@Time    :   2021-12-13 11:45:37 
@Author  :   Bingjie Yan 
@Email   :   bj.yan.pa@qq.com 
@License :   Apache License 2.0 
"""

from abc import ABC, abstractmethod
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad

from fedhf.model import build_criterion, build_optimizer


class AbsAttactor(ABC):
    def __init__(self, args):
        self.args = args

    @abstractmethod
    def attack(self):
        raise NotImplementedError


class BaseAttactor(AbsAttactor):
    def __init__(self, args):
        super(BaseAttactor, self).__init__(args)
        self.crit = build_criterion(self.args.train_loss)
        self.data_size = (self.args.batch_size, self.args.input_c, self.args.image_size,
                          self.args.image_size)
        self.label_size = (self.args.batch_size, self.args.num_classes)

    def attack(self, model, origin_grad):
        dummy_data = torch.randn(self.data_size).to(self.args.device).requires_grad_(True)
        dummy_label = torch.randn(self.label_size).to(self.args.device).requires_grad_(True)
        optimizer = torch.optim.LBFGS([dummy_data, dummy_label])

        for iters in range(300):

            def closure():
                optimizer.zero_grad()
                dummy_pred = model(dummy_data)
                dummy_loss = self.crit(dummy_pred, F.softmax(dummy_label, dim=-1))
                dummy_grad = grad(dummy_loss, model.parameters(), create_graph=True)

                grad_diff = 0
                for gx, gy in zip(dummy_grad, origin_grad):
                    grad_diff += ((gx - gy)**2).sum()
                grad_diff.backward()
                return grad_diff

            optimizer.step(closure)

        return dummy_data, dummy_label
