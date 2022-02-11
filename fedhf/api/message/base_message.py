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
    def pack(self):
        raise NotImplementedError

    @abstractmethod
    def unpack(self):
        raise NotImplementedError


class BaseMessage(AbsMessage):
    def __init__(self,
                 message_from=None,
                 message_code=mc.INFO_CODE,
                 content="",
                 message_type="",
                 *args,
                 **kwargs):
        self.message_from = message_from
        self.message_code = message_code
        self.content = deepcopy(content)
        self.message_type = message_type

    def pack(self):
        self.header = {
            "message_from": self.message_from,
            "message_code": self.message_code,
        }
        self.body = {
            "content": self.content,
            "message_type": self.message_type,
        }
        return {'header': self.header, 'body': self.body}

    def unpack(self, package):
        self.message_from = package["header"]["message_from"]
        self.message_code = package["header"]["message_code"]
        self.content = package["body"]["content"]
        self.message_type = package["body"]["message_type"]

    def __str__(self) -> str:
        return {
            "message_from": self.message_from,
            "message_code": self.message_code,
            "content": self.content,
            "message_type": self.message_type
        }.__str__()