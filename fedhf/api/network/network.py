#!/usr/bin/env python
# -*- encoding: utf-8 -*-
""" 
@File    :   fedhf\api\network\network.py 
@Time    :   2021-12-06 16:19:23 
@Author  :   Bingjie Yan 
@Email   :   bj.yan.pa@qq.com 
@License :   Apache License 2.0 
"""

import torch
import torch.distributed as dist


class Network(object):
    def __init__(self, args):
        self.addr = args.addr
        self.port = args.port
        self.world_size = args.world_size
        self.rank = args.rank
        self.backend = args.backend

    def init(self):
        dist.init_process_group(
            backend=self.backend,
            init_method="tcp://{}:{}".format(self.addr, self.port),
            world_size=self.world_size,
            rank=self.rank,
        )
