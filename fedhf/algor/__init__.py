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
from fedhf.api import EmptyConfig


def init_algor(args) -> Any:
    # register components
    algor = build_algor(args.algor)
    temp_cfg = None
    if args.get(args.algor, None) is not None:
        temp_cfg = EmptyConfig(args.get(args.algor))

    # update parameters
    args.update(algor.default_params)
    if temp_cfg is not None:
        args[args.algor].update(temp_cfg)

    for component_name, component in algor.components.items():
        register(component_name, component)

    return args
