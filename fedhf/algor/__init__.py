#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    :   fedhf\algor\__init__.py
# @Time    :   2022-07-15 12:54:38
# @Author  :   Bingjie Yan
# @Email   :   bj.yan.pa@qq.com
# @License :   Apache License 2.0

from typing import Any
from .base import build_algor, algor_factory
from fedhf.core.registor import register
from fedhf.api import Config


def register_components() -> Any:
    for algor in algor_factory.values():
        for component_name, component in algor.components.items():
            register(component_name, component)


def init_algor(args) -> Any:
    register_components()
    # register components
    algor = build_algor(args.algor)

    args.merge(algor.default_args, overwrite=False)

    return args
