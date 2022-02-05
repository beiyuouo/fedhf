#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   fedhf\component\coordinator\base_coordinator.py
@Time    :   2021-10-26 11:06:07
@Author  :   Bingjie Yan
@Email   :   bj.yan.pa@qq.com
@License :   Apache License 2.0
"""

from abc import ABC, abstractmethod

from fedhf.api import Logger
from fedhf.core import build_server, build_client
from fedhf.component import build_sampler, DistributedCommunicator
from fedhf.dataset import ClientDataset, build_dataset


class AbsCoordinator(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def prepare(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def main(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def finish(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def run(self) -> None:
        raise NotImplementedError


class SimulatedBaseCoordinator(AbsCoordinator):
    def __init__(self, args) -> None:
        super().__init__()
        self.args = args
        self.logger = Logger(self.args)

    def prepare(self) -> None:
        self.dataset = build_dataset(self.args.dataset)(self.args)
        self.sampler = build_sampler(self.args.sampler)(self.args)

        if self.args.test:
            # reduce data for test
            self.data = [
                ClientDataset(self.dataset.trainset,
                              range(i * self.args.batch_size, (i + 1) * self.args.batch_size))
                for i in range(self.args.num_clients)
            ]
        else:
            self.data = self.sampler.sample(self.dataset.trainset)

        self.client_list = [i for i in range(self.args.num_clients)]
        self.server = build_server(self.args.deploy_mode)(self.args)

    def main(self) -> None:
        pass

    def finish(self) -> None:
        self.server.model.save()

        try:
            if self.args.evaluate_on_client:
                self.logger.info("Evaluate on client")
                for client_id in self.client_list:
                    client = build_client(self.args.deploy_mode)(self.args, client_id)
                    result = client.evaluate(data=self.data[client_id], model=self.server.model)
                    self.logger.info(f'Client {client_id} result: {result}')

            result = self.server.evaluate(self.dataset.testset)
            self.logger.info(f'Server result: {result}')
            self.logger.info(
                f'Final server model version: {self.server.model.get_model_version()}')
        except KeyboardInterrupt:
            self.logger.info(f'Interrupted by user.')

        self.logger.info(f'All finished.')

    def run(self) -> None:
        self.prepare()
        self.main()
        self.finish()


class DistributedBaseCoordinator(AbsCoordinator):
    def __init__(self, args) -> None:
        super().__init__()
        self.args = args
        self.logger = Logger(self.args)
        self.communicator = DistributedCommunicator(self.args)

    def prepare(self) -> None:
        self.dataset = build_dataset(self.args.dataset)(self.args)
        self.sampler = build_sampler(self.args.sampler)(self.args)

        if self.args.test:
            # reduce data for test
            self.data = [
                ClientDataset(self.dataset.trainset,
                              range(i * self.args.batch_size, (i + 1) * self.args.batch_size))
                for i in range(self.args.num_clients)
            ]
        else:
            self.data = self.sampler.sample(self.dataset.trainset)

        self.client_list = [i for i in range(self.args.num_clients)]
        self.server = build_server(self.args.deploy_mode)(self.args)

    def main(self) -> None:
        pass

    def finish(self) -> None:
        self.server.model.save()

        try:
            if self.args.evaluate_on_client:
                self.logger.info("Evaluate on client")
                for client_id in self.client_list:
                    client = build_client(self.args.deploy_mode)(self.args, client_id)
                    result = client.evaluate(data=self.data[client_id], model=self.server.model)
                    self.logger.info(f'Client {client_id} result: {result}')

            result = self.server.evaluate(self.dataset.testset)
            self.logger.info(f'Server result: {result}')
            self.logger.info(
                f'Final server model version: {self.server.model.get_model_version()}')
        except KeyboardInterrupt:
            self.logger.info(f'Interrupted by user.')

        self.logger.info(f'All finished.')

    def run(self) -> None:
        self.prepare()
        self.main()
        self.finish()