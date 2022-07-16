#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    :   fedhf\component\sampler\base_sampler.py
# @Time    :   2022-05-03 16:00:39
# @Author  :   Bingjie Yan
# @Email   :   bj.yan.pa@qq.com
# @License :   Apache License 2.0

from abc import ABC


class BaseSampler(ABC):
    def __init__(self, args) -> None:
        self.args = args

    def sample(self):
        raise NotImplementedError
