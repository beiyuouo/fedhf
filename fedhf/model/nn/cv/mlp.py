#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    :   fedhf\model\nn\mlp.py
# @Time    :   2022-05-03 16:07:23
# @Author  :   Bingjie Yan
# @Email   :   bj.yan.pa@qq.com
# @License :   Apache License 2.0

from collections import OrderedDict
import torch
import torch.nn as nn

from fedhf import Config
from ..base_model import BaseModel


class MLPMNIST(BaseModel):
    default_args = Config(
        mlp={"input_dim": 28 * 28, "hidden_dim": 200, "output_dim": 10}
    )

    def __init__(
        self,
        args,
        **kwargs,
    ):
        super().__init__(args, **kwargs)
        self.add_default_args()

        self.net = nn.Sequential(
            OrderedDict(
                [
                    (
                        "fc1",
                        nn.Linear(self.args.mlp.input_dim, self.args.mlp.hidden_dim),
                    ),
                    ("relu1", nn.ReLU()),
                    (
                        "fc2",
                        nn.Linear(self.args.mlp.hidden_dim, self.args.mlp.hidden_dim),
                    ),
                    ("relu2", nn.ReLU()),
                    (
                        "fc3",
                        nn.Linear(self.args.mlp.hidden_dim, self.args.mlp.output_dim),
                    ),
                ]
            )
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.net(x)
        return x
