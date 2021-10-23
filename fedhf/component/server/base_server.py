#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   fedhf\component\server\base_server.py
@Time    :   2021/10/19 11:15:43
@Author  :   Bingjie Yan
@Email   :   bj.yan.pa@qq.com
"""

from abc import ABC


class Server(object):
    def __init__(self) -> None:
        super().__init__()
