#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   fedhf\component\selector\__init__.py
@Time    :   2021-11-08 21:14:03
@Author  :   Bingjie Yan
@Email   :   bj.yan.pa@qq.com
@License :   Apache License 2.0
"""

__all__ = ["build_selector"]

from .random_selector import RandomSelector

selector_factory = {
    'random': RandomSelector
}


def build_selector(sele_name: str):
    if sele_name not in selector_factory.keys():
        raise ValueError(f'{sele_name} is not a valid selector name')

    selector = selector_factory[sele_name]

    return selector
