#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   fedhf\component\trainer\base_train.py
@Time    :   2021-10-26 20:42:25
@Author  :   Bingjie Yan
@Email   :   bj.yan.pa@qq.com
@License :   Apache License 2.0
"""

from abc import ABC, abstractmethod


class BaseTrainer(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def train(self):
        raise NotImplementedError

    @abstractmethod
    def evaluate(self):
        raise NotImplementedError
