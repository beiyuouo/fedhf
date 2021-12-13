#!/usr/bin/env python
# -*- encoding: utf-8 -*-
""" 
@File    :   fedhf\core\coordinator\distributed_coordinator.py 
@Time    :   2021-12-06 13:41:54 
@Author  :   Bingjie Yan 
@Email   :   bj.yan.pa@qq.com 
@License :   Apache License 2.0 
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from .base_coordinator import DistributedBaseCoordinator


class DistributedCoordinator(DistributedBaseCoordinator):
    def __init__(self, args) -> None:
        super().__init__(args)

        assert self.network.rank == 0, "Only rank 0 can run this code"

    def prepare(self) -> None:
        super().prepare()

    def main(self) -> None:
        super().main()

    def finish(self) -> None:
        super().finish()
        if dist.is_initialized():
            dist.destroy_process_group()

    def run(self) -> None:
        self.prepare()
        self.main()
        self.finish()