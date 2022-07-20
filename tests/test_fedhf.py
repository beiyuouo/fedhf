#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    :   tests\test_fedhf.py
# @Time    :   2022-07-16 23:59:29
# @Author  :   Bingjie Yan
# @Email   :   bj.yan.pa@qq.com
# @License :   Apache License 2.0

import fedhf


class TestFedHF:
    def test_init(self):
        args = fedhf.init()
        assert args is not None

    def test_run(self):
        args = fedhf.init(debug=True, num_clients=3, num_rounds=1, num_epochs=1)
        # print(args.model)
        fedhf.run(args)
