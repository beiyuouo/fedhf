#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    :   fedhf\algor\base.py
# @Time    :   2022-07-15 21:34:03
# @Author  :   Bingjie Yan
# @Email   :   bj.yan.pa@qq.com
# @License :   Apache License 2.0


from .async_ import fedasync
from .sync import fedavg, fedprox

algor_factory = {
    "fedasync": fedasync,
    "fedavg": fedavg,
    "fedprox": fedprox,
}


def build_algor(algor_type: str):
    if algor_type not in algor_factory.keys():
        raise ValueError(f"{algor_type} is not a valid algor name")

    return algor_factory[algor_type]
