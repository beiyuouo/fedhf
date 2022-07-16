#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    :   fedhf\core\client\__init__.py
# @Time    :   2022-05-03 16:02:14
# @Author  :   Bingjie Yan
# @Email   :   bj.yan.pa@qq.com
# @License :   Apache License 2.0

__all__ = ["SimulatedClient", "build_client", "client_factory"]

from .simulated_client import SimulatedClient

client_factory = {"simulated": SimulatedClient}


def build_client(client_type):
    if client_type not in client_factory.keys():
        raise ValueError(
            "client_type {} not in {}".format(client_type, client_factory.keys())
        )
    client = client_factory[client_type]
    return client
