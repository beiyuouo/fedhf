#!/usr/bin/env python
# -*- encoding: utf-8 -*-
""" 
@File    :   fedhf\core\client\distributed_client.py 
@Time    :   2021-12-06 14:21:36 
@Author  :   Bingjie Yan 
@Email   :   bj.yan.pa@qq.com 
@License :   Apache License 2.0 
"""

import threading

import torch
import torch.nn as nn
import torch.distributed as dist
from .base_client import BaseClient


class DistributedClient(BaseClient):
    def __init__(self, args, client_id) -> None:
        super().__init__(args, client_id)
        self.status = None

    def launch(self):
        # TODO
        self.thread_train = threading.Thread(target=self.train)
        self.thread_commu = threading.Thread(target=self.communicate)