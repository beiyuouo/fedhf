#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    :   fedhf\component\selector\__init__.py
# @Time    :   2022-05-03 16:00:53
# @Author  :   Bingjie Yan
# @Email   :   bj.yan.pa@qq.com
# @License :   Apache License 2.0

__all__ = ["build_selector", "RandomFedAsyncSelector", "RandomSelector", "selector_factory"]

from .random_fedasync_selector import RandomFedAsyncSelector
from .random_selector import RandomSelector

selector_factory = {
    'random': RandomSelector,
    'random_fedasync': RandomFedAsyncSelector,
}


def build_selector(sele_name: str):
    if sele_name not in selector_factory.keys():
        raise ValueError(f'{sele_name} is not a valid selector name')

    selector = selector_factory[sele_name]

    return selector
