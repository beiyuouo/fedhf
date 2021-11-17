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
import heapq

import numpy as np

from fedhf.core import build_server, build_client

from fedhf.component import Logger, build_sampler
from fedhf.dataset import ClientDataset, build_dataset

from .base_coordinator import BaseCoordinator


class SimulatedAsyncCoordinator(BaseCoordinator):
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

        self._model_pool = []
        self._model_heap = None

    def main(self) -> None:

        for i in range(self.args.num_rounds):
            selected_client = self.server.select(self.client_list)
            self._model_pool = []
            self._model_heap = []

            self.logger.info(f'Round {i} selected client: {selected_client}')

            for client_id in selected_client:
                model = deepcopy(self.server.model)
                client = build_client('simulated')(self.args, client_id)
                model = client.train(self.data[client_id], model)

                self._model_pool.append(model)
                heapq.heappush(
                    self._model_heap,
                    (model.get_model_version() +
                     np.random.randint(0, self.args.fedasync_max_staleness),
                     np.random.rand(), model))

            while len(self._model_heap) > 0:
                _, _, model = self._model_heap.pop()

                self.server.update(model)

                result = self.server.evaluate(self.dataset.testset)
                self.logger.info(f'Server result: {result}')

        self.logger.info(f'All rounds finished.')

    def finish(self) -> None:
        for client_id in self.client_list:
            client = build_client('simulated')(self.args, client_id)
            result = client.evaluate(data=self.data[client_id],
                                     model=self.server.model)
            self.logger.info(f'Client {client_id} result: {result}')

        result = self.server.evaluate(self.dataset.testset)
        self.logger.info(f'Server result: {result}')
        self.logger.info(
            f'Final server model version: {self.server.model.get_model_version()}'
        )
        self.logger.info(f'Finish.')

    def run(self) -> None:
        self.prepare()
        self.main()
        self.finish()
