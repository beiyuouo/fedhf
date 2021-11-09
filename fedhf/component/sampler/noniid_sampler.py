#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   fedhf\component\sampler\noniid_sampler.py
@Time    :   2021-11-08 21:10:21
@Author  :   Bingjie Yan
@Email   :   bj.yan.pa@qq.com
@License :   Apache License 2.0
"""

from .base_sampler import BaseSampler


class NonIIDSampler(BaseSampler):
    def __init__(self, args) -> None:
        self.args = args

    def sample(self, dataset):
        pass
