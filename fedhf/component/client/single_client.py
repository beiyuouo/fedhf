#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   fedhf\component\client\single_client.py
@Time    :   2021-10-26 11:06:26
@Author  :   Bingjie Yan
@Email   :   bj.yan.pa@qq.com
@License :   Apache License 2.0
"""


from base_client import BaseClient


class SingleClient(BaseClient):
    def __init__(self, args) -> None:
        self.args = args
