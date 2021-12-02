#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   fedhf\component\evaluator\__init__.py
@Time    :   2021-10-26 20:47:06
@Author  :   Bingjie Yan
@Email   :   bj.yan.pa@qq.com
@License :   Apache License 2.0
"""

__all__ = ["Evaluator", "build_evalutor", "evaluator_factory"]

from .evaluator import Evaluator

evaluator_factory = {
    'evaluator': Evaluator,
}


def build_evalutor(name):
    if name not in evaluator_factory.keys():
        raise ValueError(f'Unknown evaluator name: {name}')
    return evaluator_factory[name]