#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    :   tests\test_api\test_logger.py
# @Time    :   2022-05-02 23:35:51
# @Author  :   Bingjie Yan
# @Email   :   bj.yan.pa@qq.com
# @License :   Apache License 2.0

import pytest
from fedhf import Logger
import fedhf


@pytest.mark.order(1)
class TestLogger:
    args = fedhf.init(prj_name="fedhf")

    def test_logger(self):
        self.logger = Logger(self.args)
