#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    :   fedhf\component\trainer\__init__.py
# @Time    :   2022-05-03 16:01:12
# @Author  :   Bingjie Yan
# @Email   :   bj.yan.pa@qq.com
# @License :   Apache License 2.0

__all__ = ["build_trainer", "trainer_factory", "BaseTrainer", "DefaultTrainer"]

from .base_trainer import BaseTrainer
from .default_trainer import DefaultTrainer


trainer_factory = {
    "base": BaseTrainer,
    "base_trainer": BaseTrainer,
    "default": DefaultTrainer,
    "default_trainer": DefaultTrainer,
}


def build_trainer(trainer_type: str):
    if trainer_type not in trainer_factory.keys():
        raise ValueError(f"unknown trainer type: {trainer_type}")

    return trainer_factory[trainer_type]
