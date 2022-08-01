#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    :   tests\test_api\test_utils.py
# @Time    :   2022-08-01 19:40:57
# @Author  :   Bingjie Yan
# @Email   :   bj.yan.pa@qq.com
# @License :   Apache License 2.0


import pytest
import torch
import numpy as np

import fedhf
from fedhf.api.serial.serializer import Serializer
from fedhf.model import build_model
from fedhf.api.utils.torch_utils import get_communication_cost, get_model_param_flops


@pytest.mark.order(1)
class TestUtils:
    args = fedhf.init(num_classes=10, model="cnn_mnist")

    def test_serializer_cnn(self):
        model = build_model(self.args.model)(self.args)

        serialized_model = Serializer.serialize_model(model)
        print(serialized_model.size())
        print(get_communication_cost(serialized_model))
        assert get_communication_cost(serialized_model) == 1686978

    def test_get_model_param_flops(self):
        model = build_model(self.args.model)(self.args)
        flops, params = get_model_param_flops(model, (1, 28, 28))
        print(flops, params)
        assert flops == 8482304.0
        assert params == 421642.0
