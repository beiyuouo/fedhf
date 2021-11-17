#!/usr/bin/env python
# -*- encoding: utf-8 -*-
""" 
@File    :   fedhf\model\base_model.py 
@Time    :   2021-11-15 17:48:36 
@Author  :   Bingjie Yan 
@Email   :   bj.yan.pa@qq.com 
@License :   Apache License 2.0 
"""

import time
import os

import torch
import torch.nn as nn


class BaseModel(nn.Module):
    def __init__(self, args, model_time=None, model_version=0):
        super().__init__()
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

    def save(self, path: str = None):
        if path is None:
            path = os.path.join(self.args.save_dir, f'{self.args.name}.pth')
        torch.save(
            {
                'model_version': self.model_version,
                'model_time': self.model_time,
                'state_dict': self.state_dict(),
            }, path)
