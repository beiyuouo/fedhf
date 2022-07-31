#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    :   fedhf\component\selector\__init__.py
# @Time    :   2022-05-03 16:00:53
# @Author  :   Bingjie Yan
# @Email   :   bj.yan.pa@qq.com
# @License :   Apache License 2.0

__all__ = ["build_selector", "selector_factory", "RandomSelector"]

from .random_selector import RandomSelector

selector_factory = {
    "random": RandomSelector,
}


def build_selector(sele_name: str):
    if sele_name not in selector_factory.keys():
        raise ValueError(f"unknown selector name: {sele_name}")

    selector = selector_factory[sele_name]

    return selector
