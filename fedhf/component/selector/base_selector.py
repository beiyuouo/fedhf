#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   fedhf\component\selector\base_selector.py
@Time    :   2021-11-08 21:17:01
@Author  :   Bingjie Yan
@Email   :   bj.yan.pa@qq.com
@License :   Apache License 2.0
"""


from abc import ABC


class BaseSelector(ABC):
    def __init__(self) -> None:
        super().__init__()

    def select(self):
        raise NotImplementedError
