#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    :   fedhf\component\trainer\base_trainer.py
# @Time    :   2022-05-03 16:01:18
# @Author  :   Bingjie Yan
# @Email   :   bj.yan.pa@qq.com
# @License :   Apache License 2.0

from abc import ABC, abstractmethod

from fedhf.api import Logger
from fedhf.model import build_criterion, build_optimizer, build_lr_scheduler


class AbsTrainer(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def train(self):
        raise NotImplementedError


class BaseTrainer(AbsTrainer):
    def __init__(self, args):
        self.args = args
        self.optim = build_optimizer(self.args.optim)
        self.crit = build_criterion(self.args.loss)
        if self.args.lr_scheduler:
            self.lr_scheduler = build_lr_scheduler(self.args.lr_scheduler)
        else:
            self.lr_scheduler = None
        self.logger = Logger(self.args)

    def set_device(self, gpus, device):
        if len(gpus) > 1:
            pass
        else:
            pass

    def train(self):
        print("BaseTrainer")
