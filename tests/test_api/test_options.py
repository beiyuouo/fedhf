#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    :   tests\test_api\test_options.py
# @Time    :   2022-03-08 21:19:51
# @Author  :   Bingjie Yan
# @Email   :   bj.yan.pa@qq.com
# @License :   Apache License 2.0

from fedhf.api import opts


class TestOptions:

    def test_get_opt(self):
        args = opts().parse(args=['--lr', '0.1'])
        assert args.lr == 0.1