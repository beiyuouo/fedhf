#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   fedhf\component\coordinator\single_coordinator.py
@Time    :   2021-10-26 11:06:00
@Author  :   Bingjie Yan
@Email   :   bj.yan.pa@qq.com
@License :   Apache License 2.0
"""

from copy import deepcopy
from fedhf.component import logger

from fedhf.core.server import build_server
from fedhf.core.client import build_client

from fedhf.component.aggregator import build_aggregator
from fedhf.component.logger import Logger
from fedhf.component.sampler import build_sampler
from fedhf.component.selector import build_selector
from fedhf.dataset.client_datasest import ClientDataset
from fedhf.model import build_criterion, build_model, build_optimizer
from fedhf.dataset import build_dataset

from .base_coordinator import BaseCoordinator


class SimulatedCoordinator(BaseCoordinator):
    """Simulated Coordinator
        In simulated scheme, the data and model belong to coordinator and there is no need communicator.
        Also, there is no need to instantiate every client.
    """
    def __init__(self, args) -> None:
        self.args = args

    def prepare(self) -> None:
        self.dataset = build_dataset(self.args.dataset)(self.args)
        self.sampler = build_sampler(self.args.sampler)(self.args)

        if self.args.test:
            # reduce data for test
            self.data = [
                ClientDataset(
                    self.dataset.trainset,
                    range(i * self.args.batch_size,
                          (i + 1) * self.args.batch_size))
                for i in range(self.args.num_clients)
            ]
        else:
            self.data = self.sampler.sample(self.dataset.trainset)

        self.client_list = [i for i in range(self.args.num_clients)]
        self.server = build_server('simulated')(self.args)
        # self.client = build_client('simulated')(self.args)

        self.logger = Logger(self.args)

    def main(self) -> None:
        for i in range(self.args.num_rounds):
            selected_client = self.server.select(self.client_list)
            for client_id in selected_client:
                model = deepcopy(self.server.model)
                client = build_client('simulated')(self.args, client_id)
                model = client.train(self.data[client_id], model)
                self.server.update(model)

    def finish(self) -> None:
        for client_id in self.client_list:
            client = build_client('simulated')(self.args, client_id)
            result = client.evaluate(data=self.data[client_id],
                                     model=self.server.model)
            self.logger.info(f'Client {client_id} result: {result}')

        result = self.server.evaluate(self.dataset.testset)
        self.logger.info(f'Server result: {result}')

    def run(self) -> None:
        self.prepare()
        self.main()
        self.finish()
