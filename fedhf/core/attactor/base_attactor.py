#!/usr/bin/env python
# -*- encoding: utf-8 -*-
""" 
@File    :   fedhf\core\attactor\base_attactor.py 
@Time    :   2021-12-13 11:45:37 
@Author  :   Bingjie Yan 
@Email   :   bj.yan.pa@qq.com 
@License :   Apache License 2.0 
"""

from abc import ABC, abstractmethod


class AbsAttactor(ABC):
    def __init__(self, args):
        self.args = args

    @abstractmethod
    def attack(self):
        raise NotImplementedError


class BaseAttactor(AbsAttactor):
    def __init__(self, args):
        super(BaseAttactor, self).__init__(args)

    @staticmethod
    def attack(model, gradient):
        raise NotImplementedError