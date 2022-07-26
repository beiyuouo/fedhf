#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    :   fedhf\component\evaluator\__init__.py
# @Time    :   2022-05-03 16:00:21
# @Author  :   Bingjie Yan
# @Email   :   bj.yan.pa@qq.com
# @License :   Apache License 2.0

__all__ = ["build_evaluator", "evaluator_factory", "BaseEvaluator", "DefaultEvaluator"]

from .base_evaluator import BaseEvaluator
from .default_evaluator import DefaultEvaluator


evaluator_factory = {
    "base": BaseEvaluator,
    "base_evaluator": BaseEvaluator,
    "default": DefaultEvaluator,
    "default_evaluator": DefaultEvaluator,
}


def build_evaluator(name):
    if name not in evaluator_factory.keys():
        raise ValueError(f"unknown evaluator name: {name}")
    return evaluator_factory[name]
