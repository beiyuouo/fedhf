#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    :   fedhf\component\evaluator\base_evaluator.py
# @Time    :   2022-05-03 16:00:25
# @Author  :   Bingjie Yan
# @Email   :   bj.yan.pa@qq.com
# @License :   Apache License 2.0

from abc import ABC, abstractmethod

from fedhf.model import build_criterion, build_optimizer
from fedhf.api import Logger


class AbsEvaluator(ABC):

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def evaluate(self):
        raise NotImplementedError


class BaseEvaluator(AbsEvaluator):

    def __init__(self, args) -> None:
        self.args = args
        self.crit = build_criterion(self.args.loss)
        self.logger = Logger(self.args)

    def set_device(self, gpus, device):
        if len(gpus) > 1:
            pass
        else:
            pass

    def evaluate(self):
        pass