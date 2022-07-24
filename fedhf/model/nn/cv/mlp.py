#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    :   fedhf\model\nn\mlp.py
# @Time    :   2022-05-03 16:07:23
# @Author  :   Bingjie Yan
# @Email   :   bj.yan.pa@qq.com
# @License :   Apache License 2.0

import torch
import torch.nn as nn

from fedhf import Config
from ..base_model import BaseModel


class MLP(BaseModel):
    default_args = Config(
        mlp={"input_dim": 28 * 28, "hidden_dim": 128, "output_dim": 10}
    )

    def __init__(
        self,
        args,
        **kwargs,
    ):
        super().__init__(args, **kwargs)
        self.add_default_args()

        self.layer_input = nn.Linear(
            self.args.mlp.get("input_dim"), self.args.mlp.get("hidden_dim")
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.layer_hidden = nn.Linear(
            self.args.mlp.get("hidden_dim"), self.args.mlp.get("output_dim")
        )

    def forward(self, x):
        x = x.view(-1, x.shape[1] * x.shape[-2] * x.shape[-1])
        x = self.layer_input(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.layer_hidden(x)
        return x
