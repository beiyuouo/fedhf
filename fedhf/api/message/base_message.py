#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   fedhf\api\message\base_message.py
@Time    :   2021-10-26 11:07:18
@Author  :   Bingjie Yan
@Email   :   bj.yan.pa@qq.com
@License :   Apache License 2.0
"""


from abc import ABC, abstractmethod


class BaseMessage(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def send(self, *args, **kwargs):
        raise NotImplementedError
