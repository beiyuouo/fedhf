#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   fedhf\component\aggregator\base_aggregator.py
@Time    :   2021-10-26 20:33:11
@Author  :   Bingjie Yan
@Email   :   bj.yan.pa@qq.com
@License :   Apache License 2.0
"""

from abc import ABC, abstractmethod


class BaseAggregator(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def agg(self):
        raise NotImplementedError

    @abstractmethod
    def _check_agg(self):
        raise NotImplementedError
