#!/usr/bin/env python
# -*- encoding: utf-8 -*-
""" 
@File    :   tests\test_component\test_serializer.py 
@Time    :   2021-11-15 18:24:08 
@Author  :   Bingjie Yan 
@Email   :   bj.yan.pa@qq.com 
@License :   Apache License 2.0 
"""
import numpy as np
import torch
from torch.utils.data import DataLoader

from fedhf.api import opts
from fedhf.component.serializer import Serializer, Deserializer
from fedhf.model import build_model


class TestSerializer(object):
    args = opts().parse(['--num_classes', '10', '--model', 'resnet'])

    def test_serializer(self):
        model = build_model(self.args.model)(self.args)

        serialized_model = Serializer.serialize_model(model)
        print(serialized_model.size())
        total_params = 0
        for param in model.parameters():
            total_params += np.prod(param.size())
        print(total_params)
        assert type(serialized_model) == torch.Tensor
        assert serialized_model.size()[0] == total_params

        model_ = build_model(self.args.model)(self.args)
        Deserializer.deserialize_model(model_, serialized_model)

        for param1_kv, param2_kv in zip(model.parameters(),
                                        model_.parameters()):
            param1 = param1_kv[1]
            param2 = param2_kv[1]
            # print(param1_kv[0], param2_kv[0])
            assert torch.all(param1 == param2)