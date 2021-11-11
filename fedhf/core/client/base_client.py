#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   fedhf\component\client\base_client.py
@Time    :   2021-10-26 11:06:33
@Author  :   Bingjie Yan
@Email   :   bj.yan.pa@qq.com
@License :   Apache License 2.0
"""


from abc import ABC, abstractmethod


class BaseClient(ABC):
    def __init__(self) -> None:
        super().__init__()
