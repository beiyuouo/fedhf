#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    :   fedhf\algor\__init__.py
# @Time    :   2022-07-15 12:54:38
# @Author  :   Bingjie Yan
# @Email   :   bj.yan.pa@qq.com
# @License :   Apache License 2.0

from typing import Any
from .base import build_algor
from fedhf.core.registor import register


def init_algor(args) -> Any:
    # register components
    algor = build_algor(args.algor)

    # update parameters
    algor.default_params.update(args)
    args = algor.default_params

    for component_name, component in algor.components.items():
        register(component_name, component)

    return args
