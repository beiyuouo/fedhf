#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    :   fedhf\component\selector\base_selector.py
# @Time    :   2022-05-03 16:00:57
# @Author  :   Bingjie Yan
# @Email   :   bj.yan.pa@qq.com
# @License :   Apache License 2.0

from abc import ABC


class BaseSelector(ABC):

    def __init__(self) -> None:
        super().__init__()

    def select(self):
        raise NotImplementedError
