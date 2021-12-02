#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   fedhf\component\trainer\__init__.py
@Time    :   2021-10-26 20:43:41
@Author  :   Bingjie Yan
@Email   :   bj.yan.pa@qq.com
@License :   Apache License 2.0
"""

__all__ = ["Trainer", "FedAsyncTrainer", "build_trainer", "trainer_factory"]

from .trainer import Trainer
from .fedasync_trainer import FedAsyncTrainer

trainer_factory = {
    'trainer': Trainer,
    'fedasync_trainer': FedAsyncTrainer,
}


def build_trainer(trainer_type: str):
    if trainer_type not in trainer_factory.keys():
        raise ValueError(f'{trainer_type} is not a valid trainer name')

    return trainer_factory[trainer_type]
