#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   fedhf\component\evaluator\base_evaluator.py
@Time    :   2021-10-26 20:47:02
@Author  :   Bingjie Yan
@Email   :   bj.yan.pa@qq.com
@License :   Apache License 2.0
"""


from abc import ABC, abstractmethod


class BaseEvaluator(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def evaluate(self):
        raise NotImplementedError
