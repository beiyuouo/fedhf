#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   fedhf\component\client\__init__.py
@Time    :   2021-10-26 11:06:38
@Author  :   Bingjie Yan
@Email   :   bj.yan.pa@qq.com
@License :   Apache License 2.0
"""


__all__ = []

from .simulated_client import SimulatedClient

client_factory = {
    'simluated': SimulatedClient
}

def build_client(client_type):
    if client_type not in client_factory.keys():
        raise ValueError('client_type {} not in {}'.format(client_type, client_factory.keys()))
    client = client_factory[client_type]
    return client