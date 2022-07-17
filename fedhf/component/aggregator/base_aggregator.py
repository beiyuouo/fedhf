#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    :   fedhf\component\aggregator\base_aggregator.py
# @Time    :   2022-05-03 15:59:51
# @Author  :   Bingjie Yan
# @Email   :   bj.yan.pa@qq.com
# @License :   Apache License 2.0

from abc import ABC, abstractmethod
from fedhf.api import Logger


class AbsAggregator(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def agg(self):
        raise NotImplementedError

    @abstractmethod
    def _check_agg(self):
        raise NotImplementedError


class BaseAggregator(AbsAggregator):
    def __init__(self, args) -> None:
        self.args = args
        self.logger = Logger(self.args)

    def agg(self):
        pass

    def _check_agg(self):
        pass
