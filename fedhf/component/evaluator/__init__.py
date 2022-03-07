#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   fedhf\component\evaluator\__init__.py
@Time    :   2021-10-26 20:47:06
@Author  :   Bingjie Yan
@Email   :   bj.yan.pa@qq.com
@License :   Apache License 2.0
"""

__all__ = ["Evaluator", "build_evaluator", "evaluator_factory", "BaseEvaluator"]

from .evaluator import Evaluator
from .base_evaluator import BaseEvaluator

evaluator_factory = {'evaluator': Evaluator, 'base_evaluator': BaseEvaluator}


def build_evaluator(name):
    if name not in evaluator_factory.keys():
        raise ValueError(f'Unknown evaluator name: {name}')
    return evaluator_factory[name]