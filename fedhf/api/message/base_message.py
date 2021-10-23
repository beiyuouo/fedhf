#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   fedhf\api\message\message.py
@Time    :   2021/10/19 21:47:14
@Author  :   Bingjie Yan
@Email   :   bj.yan.pa@qq.com
"""

from abc import ABC, abstractmethod


class BaseMessage(ABC):

    @abstractmethod
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def send(self, *args, **kwargs):
        raise NotImplementedError
