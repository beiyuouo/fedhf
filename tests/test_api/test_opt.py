#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   tests\test_api\test_opt.py
@Time    :   2021-11-10 17:30:20
@Author  :   Bingjie Yan
@Email   :   bj.yan.pa@qq.com
@License :   Apache License 2.0
"""

from fedhf.api import opts

class TestOpts:
    def test_get_opt(self):
        args = opts().parse(['--num_clients', '10'])

        assert args.num_clients == 10