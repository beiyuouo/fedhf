#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    :   fedhf\api\opt\cfg.py
# @Time    :   2022-07-04 14:06:07
# @Author  :   Bingjie Yan
# @Email   :   bj.yan.pa@qq.com
# @License :   Apache License 2.0

import os
from pathlib import Path
import ezkfg as ez
import numpy as np
import torch

from .opts import opts
from ..path import *


class Config(ez.Config):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.__parent__ = None
        self.__key__ = None

        opt = opts().parse([])  # default opts
        self.load(opt)

        cfg = None
        if opt.cfg or self.cfg:  # load cfg file
            cfg = self.cfg if self.cfg else opt.cfg
            self.load_from_file(cfg)

        # print("args:", args)
        # print("kwargs:", kwargs)
        self.load_args_kwargs(*args, **kwargs)  # load args and kwargs, highest priority

        if self.cfg and cfg != self.cfg:
            self.load_from_file(self.cfg)

        self.parse_cfg()

    def parse_cfg(self):
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        _prj_name = f"{self.scheme}_{self.num_clients}_{self.num_rounds}_{self.num_epochs}_{self.seed}"

        self.prj_name = self.prj_name if self.prj_name else _prj_name

        # make dirs
        self.save_dir = Path(self.save_dir) / self.exp_name
        self.save_dir = increment_path(self.save_dir, exist_ok=self.exist_ok, mkdir=True)
        # self.save_dir.mkdir(parents=True, exist_ok=True)

        # logger
        if self.log_dir:
            self.log_dir = Path(self.log_dir) / self.exp_name
            self.log_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.log_dir = self.save_dir / "log"
            self.log_dir.mkdir(parents=True, exist_ok=True)

        if self.log_file is None:
            self.log_file = self.log_dir / f"{self.exp_name}.log"

        # device
        self.gpus_str = str(self.gpus)
        self.gpus = [int(gpu) for gpu in self.gpus.split(",")]
        self.gpus = [i for i in range(len(self.gpus))] if self.gpus[0] >= 0 else [-1]
        self.device = torch.device("cuda" if self.gpus[0] >= 0 else "cpu")
        if self.device != "cpu":
            torch.backends.cudnn.benchmark = True

        self.num_workers = max(self.num_workers, 2 * len(self.gpus))

        if self.scheme == "async":
            self.select_ratio = 1.0

        # print(self.get("dp"))
        if self.get("dp"):
            self.dp.epsilon = self.dp.epsilon / (self.select_ratio * self.num_epochs)

        self.num_clients_per_round = (
            self.num_clients_per_round if self.num_clients_per_round else int(self.num_clients * self.select_ratio)
        )
        self.num_clients_per_round = max(self.num_clients_per_round, 1)

        # if opt.resume and opt.load_model == "":
        #     opt.load_model = os.path.join(opt.save_dir, f"{opt.name}.pth")
