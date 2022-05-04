#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    :   fedhf\core\server\__init__.py
# @Time    :   2022-05-03 15:45:20
# @Author  :   Bingjie Yan
# @Email   :   bj.yan.pa@qq.com
# @License :   Apache License 2.0

__all__ = ["SimulatedServer", "build_server", "server_factory"]

from .simulated_server import SimulatedServer

server_factory = {
    "simulated": SimulatedServer,
}


def build_server(server_type: str):
    if server_type not in server_factory:
        raise ValueError("Unknown server type: {}".format(server_type))
    server = server_factory[server_type]
    return server
