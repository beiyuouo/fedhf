#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   fedhf\component\server\base_server.py
@Time    :   2021-10-26 11:07:00
@Author  :   Bingjie Yan
@Email   :   bj.yan.pa@qq.com
@License :   Apache License 2.0
"""


from abc import ABC


class BaseServer(ABC):
    def __init__(self) -> None:
        super().__init__()

    def update(self):
        raise NotImplementedError

    def select(self):
        raise NotImplementedError
