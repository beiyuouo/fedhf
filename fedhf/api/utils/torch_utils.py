#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    :   fedhf\api\utils\torch_utils.py
# @Time    :   2022-08-01 17:53:25
# @Author  :   Bingjie Yan
# @Email   :   bj.yan.pa@qq.com
# @License :   Apache License 2.0

import torch
import torch.nn as nn
from thop import profile
import pickle


def get_model_param_flops(model, input_size=(3, 224, 224)):
    # input_size: (channels, height, width)
    # FLOPs：FLoating point OPerationS
    # MACs：Multiply-Accumulate OperationS
    macs, params = profile(model, inputs=(torch.randn(1, *input_size),))
    return macs * 2, params


def get_communication_cost(obj) -> float:
    # return the communication cost of the object calculated by the binary size of the object
    # the unit is MB
    return len(pickle.dumps(obj))


def model_merge_ignore_bn(model1: nn.Module, model2: nn.Module) -> nn.Module:
    """
    Copy the parameters from model2 to model1, skip the batchnorm layers
    """
    for (name1, param1), (name2, param2) in zip(
        model1.named_parameters(), model2.named_parameters()
    ):
        assert name1 == name2

    for (name1, param1), (name2, param2) in zip(
        model1.named_parameters(), model2.named_parameters()
    ):
        if "bn" in name1:
            continue
        param1.data.copy_(param2.data)

    return model1
