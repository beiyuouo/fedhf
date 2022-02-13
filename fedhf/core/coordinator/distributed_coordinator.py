#!/usr/bin/env python
# -*- encoding: utf-8 -*-
""" 
@File    :   fedhf\core\coordinator\distributed_coordinator.py 
@Time    :   2021-12-06 13:41:54 
@Author  :   Bingjie Yan 
@Email   :   bj.yan.pa@qq.com 
@License :   Apache License 2.0 
"""

import threading

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from fedhf.api import mc

from .base_coordinator import DistributedBaseCoordinator


class DistributedCoordinator(DistributedBaseCoordinator):
    def __init__(self, args) -> None:
        super().__init__(args)

        assert self.network.rank == 0, "Only rank 0 can run this code"

    def prepare(self) -> None:
        super().prepare()
        # TODO: check status of all workers

    def main(self) -> None:
        round = 0

        # if coordinator and server are on the same machine, you just need to launch coordinator
        # self.server.launch()

        self.communicator.send(mc.REQUEST_CODE, 1)  # fixme
        msg = self.communicator.recv()

        while round < self.args.rounds:
            round += 1
            self.logger.info("Round {} start.".format(round))

            msg = self.communicator.recv()
            # TODO: check message code

            self.logger.info("Round {} end.".format(round))

    def finish(self) -> None:
        super().finish()

    def run(self) -> None:
        self.prepare()
        self.main()
        self.finish()