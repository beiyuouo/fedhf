#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    :   tests\test_api\test_serial.py
# @Time    :   2022-02-07 12:35:07
# @Author  :   Bingjie Yan
# @Email   :   bj.yan.pa@qq.com
# @License :   Apache License 2.0

import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader

import fedhf
from fedhf import Config, Serializer, Deserializer
from fedhf.model import build_model


class TestSerializer(object):
    args = fedhf.init(num_classes=10, model="resnet")

    def test_serializer_resnet(self):
        self.args.model = "resnet"
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

        for param1_kv, param2_kv in zip(model.parameters(), model_.parameters()):
            param1 = param1_kv[1]
            param2 = param2_kv[1]
            # print(param1_kv[0], param2_kv[0])
            assert torch.all(param1 == param2)

        assert model.get_model_version() == model_.get_model_version()

    def test_serializer_cnn(self):
        self.args.model = "cnn_mnist"
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

        for param1_kv, param2_kv in zip(model.parameters(), model_.parameters()):
            assert torch.all(param1_kv == param2_kv)

        assert model.get_model_version() == model_.get_model_version()
