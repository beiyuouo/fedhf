#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    :   fedhf\model\nn\__init__.py
# @Time    :   2022-05-03 16:06:57
# @Author  :   Bingjie Yan
# @Email   :   bj.yan.pa@qq.com
# @License :   Apache License 2.0

from .base_model import BaseModel
from .cv import *

model_factory = {}
model_factory.update(cv_model_factory)


def build_model(model_name: str):
    if model_name not in model_factory.keys():
        raise ValueError(f"unknown model name: {model_name}")
    model = model_factory[model_name]
    return model
