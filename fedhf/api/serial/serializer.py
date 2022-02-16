#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   fedhf\component\serial\serializer.py
@Time    :   2021-11-10 16:54:44
@Author  :   Bingjie Yan
@Email   :   bj.yan.pa@qq.com
@License :   Apache License 2.0
"""

import torch


class Serializer(object):
    """
    Serializer
    """
    @staticmethod
    def serialize_model_grad(model: torch.nn.Module) -> torch.Tensor:
        grads = [param.data.grad.view(-1) for param in model.parameters()]
        grads = torch.cat(grads)
        grads = grads.cpu()
        return grads

    @staticmethod
    def serialize_model(model: torch.nn.Module) -> torch.Tensor:
        params = [param.data.view(-1) for param in model.parameters()]
        params = torch.cat(params)
        params = params.cpu()
        return params