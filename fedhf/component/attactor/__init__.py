#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    :   fedhf\component\attactor\__init__.py
# @Time    :   2022-07-15 13:00:23
# @Author  :   Bingjie Yan
# @Email   :   bj.yan.pa@qq.com
# @License :   Apache License 2.0

from .base_attactor import BaseAttactor

attactor_factory = {
    "base_attactor": BaseAttactor,
}
