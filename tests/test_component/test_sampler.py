#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   tests\test_component\sampler.py
@Time    :   2021-11-09 16:06:41
@Author  :   Bingjie Yan
@Email   :   bj.yan.pa@qq.com
@License :   Apache License 2.0
"""


from fedhf.component.sampler import build_sampler
from fedhf.api import opts

class TestSampler(object):
    client_list = [i for i in range(10)]
    args = opts.parse()

    def test_random_sampler(self):
        sampler = build_sampler('random')(self.args)
        assert sampler is not None
        assert sampler.__class__.__name__ == 'RandomSampler'
        assert sampler.sample(self.client_list) is not None
        assert len(sampler.sample(self.client_list)) == 10
