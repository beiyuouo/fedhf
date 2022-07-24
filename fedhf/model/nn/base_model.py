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
    def __init__(self, args=None, **kwargs):
        super(BaseModel, self).__init__()
        self.args = args if args else Config()
        kwargs = Config(**kwargs)
        kwargs["model_time"] = kwargs.get("model_time", time.time())
        kwargs["model_version"] = kwargs.get("model_version", 0)
        self.args.merge(Config(**kwargs), overwrite=True)

    def get_model_time(self):
        return self.args.get("model_time")

    def set_model_time(self, model_time):
        self.args.set("model_time", model_time)

    def get_model_version(self):
        return self.args.get("model_version")

    def set_model_version(self, model_version):
        self.args.set("model_version", model_version)

    def save(self, path: str = None) -> None:
        if path is None:
            path = os.path.join(self.args.save_dir, f"{self.args.exp_name}.pth")
        torch.save(
            obj={
                "args": self.args,
                "state_dict": self.state_dict(),
            },
            f=path,
        )

    def load(self, path: str = None) -> None:
        if path is None:
            path = os.path.join(self.args.save_dir, f"{self.args.exp_name}.pth")
        checkpoint = torch.load(path)

        self.args = checkpoint["args"]
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
