#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    :   fedhf\component\evaluator\__init__.py
# @Time    :   2022-05-03 16:00:21
# @Author  :   Bingjie Yan
# @Email   :   bj.yan.pa@qq.com
# @License :   Apache License 2.0

__all__ = ["Evaluator", "build_evaluator", "evaluator_factory", "BaseEvaluator"]

from .evaluator import Evaluator
from .base_evaluator import BaseEvaluator

evaluator_factory = {'evaluator': Evaluator, 'base_evaluator': BaseEvaluator}


def build_evaluator(name):
    if name not in evaluator_factory.keys():
        raise ValueError(f'Unknown evaluator name: {name}')
    return evaluator_factory[name]