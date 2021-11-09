#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   tests\test_component\test_sampler.py
@Time    :   2021-11-09 20:03:30
@Author  :   Bingjie Yan
@Email   :   bj.yan.pa@qq.com
@License :   Apache License 2.0
"""

from fedhf.component.sampler import build_sampler
from fedhf.api import opts

class TestSampler(object):
    args = opts().parse()

    def test_random_sampler(self):
        sampler = build_sampler('random')(self.args)

        assert sampler is not None
        assert sampler.__class__.__name__ == 'RandomSampler'