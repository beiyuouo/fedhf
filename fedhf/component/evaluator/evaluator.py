#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   fedhf\component\evaluator\evaluator.py
@Time    :   2021-10-26 20:47:11
@Author  :   Bingjie Yan
@Email   :   bj.yan.pa@qq.com
@License :   Apache License 2.0
"""


from .base_evaluator import BaseEvaluator


class Evaluator(BaseEvaluator):
    def __init__(self, args) -> None:
        self.args = args

    def evaluate(self, data, model):
        pass
