#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   fedhf\component\server\__init__.py
@Time    :   2021-10-26 11:06:56
@Author  :   Bingjie Yan
@Email   :   bj.yan.pa@qq.com
@License :   Apache License 2.0
"""

__all__ = ["SimulatedServer", "build_server"]

from .simulated_server import SimulatedServer

server_factory = {"simulated": SimulatedServer}


def build_server(server_type: str):
    if server_type not in server_factory:
        raise ValueError("Unknown server type: {}".format(server_type))
    server = server_factory[server_type]
    return server
