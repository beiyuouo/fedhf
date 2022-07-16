#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    :   fedhf\__init__.py
# @Time    :   2022-02-13 14:32:13
# @Author  :   Bingjie Yan
# @Email   :   bj.yan.pa@qq.com
# @License :   Apache License 2.0

# import utils api
from .api import *
from .algor import *
from .core import register, register_all


def init(*args, **kwargs):
    args = Config(*args, **kwargs)
    args = init_algor(args)
    return args


def run(args=None):
    if args is None:
        args = init()

    from .core import build_coordinator

    coordinator_type = args.get("coordinator", f"{args.deploy_mode}_{args.scheme}")
    coordinator = build_coordinator(coordinator_type)(args)
    coordinator.run()
