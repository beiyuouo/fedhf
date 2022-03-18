#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   fedhf\model\mlp.py
@Time    :   2021-10-28 15:53:56
@Author  :   Bingjie Yan
@Email   :   bj.yan.pa@qq.com
@License :   Apache License 2.0
"""

import torch
import torch.nn as nn

from .base_model import BaseModel


class MLP(BaseModel):

    def __init__(self,
                 args,
                 model_time=None,
                 model_version=0,
                 input_dim=28 * 28,
                 hidden_dim=128,
                 output_dim=10):
        super().__init__(args, model_time, model_version)
        self.layer_input = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.layer_hidden = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = x.view(-1, x.shape[1] * x.shape[-2] * x.shape[-1])
        x = self.layer_input(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.layer_hidden(x)
        return x
