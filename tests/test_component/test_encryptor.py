#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    :   tests\test_component\test_encryptor.py
# @Time    :   2022-05-03 12:16:15
# @Author  :   Bingjie Yan
# @Email   :   bj.yan.pa@qq.com
# @License :   Apache License 2.0

import pytest
import fedhf
from fedhf.component import build_encryptor
from fedhf.model.nn import MLP


@pytest.mark.order(3)
class TestEncryptor:
    """
    TestEncryptor is the class for testing the encryptor.
    """

    args = fedhf.init(
        lr=0.1,
        dp={"mechanism": "none", "clip": 0.1, "epsilon": 0.1, "delta": 0.1},
        gpus="-1",
    )

    def test_none_encryptor(self):
        """
        Test the encryptor.
        """
        encryptor = build_encryptor("none")(self.args)

        model = MLP(None, mlp={"input_dim": 10 * 10, "output_dim": 10})

        encryptor.encrypt_model(model)

    def test_dp_encryptor(self):
        """
        Test the encryptor.
        """
        encryptor = build_encryptor("dp")(self.args, data_size=100)

        model = MLP(None, mlp={"input_dim": 10 * 10, "output_dim": 10})

        encryptor.encrypt_model(model)
