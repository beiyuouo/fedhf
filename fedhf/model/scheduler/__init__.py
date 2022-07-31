#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    :   fedhf\model\scheduler\__init__.py
# @Time    :   2022-02-25 20:44:28
# @Author  :   Bingjie Yan
# @Email   :   bj.yan.pa@qq.com
# @License :   Apache License 2.0

import torch
import torch.nn as nn
import torch.optim as optim

lr_schedule_factory = {
    "step": optim.lr_scheduler.StepLR,
    "multi_step": optim.lr_scheduler.MultiStepLR,
    "exponential": optim.lr_scheduler.ExponentialLR,
    "cosine": optim.lr_scheduler.CosineAnnealingLR,
}


def build_lr_scheduler(lr_schedule_name: str):
    if lr_schedule_name not in lr_schedule_factory.keys():
        raise ValueError(f"unknown lr_schedule name: {lr_schedule_name}")

    lr_scheduler = lr_schedule_factory[lr_schedule_name]
    return lr_scheduler
