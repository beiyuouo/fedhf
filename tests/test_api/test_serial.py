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

from fedhf.api import opts, Serializer, Deserializer, Message
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

        for param1_kv, param2_kv in zip(model.parameters(), model_.parameters()):
            param1 = param1_kv[1]
            param2 = param2_kv[1]
            # print(param1_kv[0], param2_kv[0])
            assert torch.all(param1 == param2)

        assert model.get_model_version() == model_.get_model_version()

    def test_unpickler(self):
        obj = Message(message_from="test", content="test")
        obj_packed = obj.pack()
        print(obj)

        buf = pickle.dumps(obj_packed)

        rev = Deserializer.load(buf)

        obj_ = Message()
        obj_.unpack(rev)

        assert obj_.message_from == obj.message_from
        assert obj_.content == obj.content
        assert obj_.message_code == obj.message_code
        assert obj_.message_type == obj.message_type
