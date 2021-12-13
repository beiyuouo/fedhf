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
from copy import deepcopy

from . import message_code as mc


class AbsMessage(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def send(self):
        raise NotImplementedError


class BaseMessage(AbsMessage):
    def __init__(self,
                 message_from,
                 message_code=mc.INFO_CODE,
                 content="",
                 dtype="",
                 *args,
                 **kwargs):
        self.message_from = message_from
        self.message_code = message_code
        self.content = deepcopy(content)
        self.dtype = dtype

    def pack(self):
        self.header = {
            "message_from": self.message_from,
            "message_code": self.message_code,
        }
        self.body = {
            "content": self.content,
            "dtype": self.dtype,
        }

    def unpack(self):
        self.message_from = self.header["message_from"]
        self.message_code = self.header["message_code"]
        self.content = self.body["content"]
        self.dtype = self.body["dtype"]

    def __str__(self) -> str:
        return {
            "message_code": self.message_code,
            "content": self.content,
            "dtype": self.dtype
        }.__str__()