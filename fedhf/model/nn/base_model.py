#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    :   fedhf\model\nn\base_model.py
# @Time    :   2022-05-03 16:07:05
# @Author  :   Bingjie Yan
# @Email   :   bj.yan.pa@qq.com
# @License :   Apache License 2.0

from copy import deepcopy
import time
import os
from typing import Any

import torch
import torch.nn as nn

from fedhf import Config


class BaseModel(nn.Module):
    def __init__(self, args, model_time=None, model_version=0):
        super(BaseModel, self).__init__()
        self.args = args
        self.model_time = model_time if model_time else time.time()
        self.model_version = model_version

    def get_model_time(self):
        return self.model_time

    def set_model_time(self, model_time):
        self.model_time = model_time

    def get_model_version(self):
        return self.model_version

    def set_model_version(self, model_version):
        self.model_version = model_version

    def save(self, path: str = None) -> None:
        if path is None:
            path = os.path.join(self.args.save_dir, f"{self.args.exp_name}.pth")
        torch.save(
            obj={
                "model_version": self.model_version,
                "model_time": self.model_time,
                "state_dict": self.state_dict(),
            },
            f=path,
        )

    def load(self, path: str = None) -> None:
        if path is None:
            path = os.path.join(self.args.save_dir, f"{self.args.exp_name}.pth")
        checkpoint = torch.load(path)
        self.model_version = checkpoint["model_version"]
        self.model_time = checkpoint["model_time"]
        self.load_state_dict(checkpoint["state_dict"])

    def add_default_args(self, args=None) -> Any:
        if args is None:
            if not hasattr(self, "default_args"):
                args = Config()
            else:
                args = deepcopy(self.default_args)
        print("func args:", args)
        self.args.merge(args, overwrite=False)
        print("func args:", self.args)
        # return self.args
