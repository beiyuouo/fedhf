#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    :   tests\test_component\test_selector.py
# @Time    :   2022-07-15 15:45:33
# @Author  :   Bingjie Yan
# @Email   :   bj.yan.pa@qq.com
# @License :   Apache License 2.0

import pytest
from fedhf import Config
import fedhf
from fedhf.component import build_selector


@pytest.mark.order(3)
class TestSelector(object):
    client_list = [i for i in range(10)]
    args = fedhf.init(num_clients=10, select_ratio=0.5)

    def test_random_selector(self):
        selector = build_selector("random")(self.args)

        assert selector is not None
        assert selector.__class__.__name__ == "RandomSelector"
        assert selector.select(self.client_list) is not None
        assert len(selector.select(self.client_list)) == int(
            self.args.num_clients * self.args.select_ratio
        )
