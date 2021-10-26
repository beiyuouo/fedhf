#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   fedhf\component\server\single_server.py
@Time    :   2021-10-26 11:07:05
@Author  :   Bingjie Yan
@Email   :   bj.yan.pa@qq.com
@License :   Apache License 2.0
"""


from base_server import BaseServer


class SingleServer(BaseServer):
    def __init__(self, args) -> None:
        self.args = args
