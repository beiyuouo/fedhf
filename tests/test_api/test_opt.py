#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    :   tests\test_api\test_opt.py
# @Time    :   2022-05-02 23:36:11
# @Author  :   Bingjie Yan
# @Email   :   bj.yan.pa@qq.com
# @License :   Apache License 2.0

from fedhf.api import opts


class TestOpts:

    def test_get_opt(self):
        args = opts().parse(['--num_clients', '10'])

        assert args.num_clients == 10