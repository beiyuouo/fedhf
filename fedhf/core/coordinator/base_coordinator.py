#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    :   fedhf\core\coordinator\base_coordinator.py
# @Time    :   2022-05-03 15:41:08
# @Author  :   Bingjie Yan
# @Email   :   bj.yan.pa@qq.com
# @License :   Apache License 2.0

from abc import ABC, abstractmethod
from fedhf.api import Logger
from fedhf.core import build_server, build_client
from fedhf.component import build_sampler
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

    def setup_data(self) -> None:
        # setup data for client
        if self.args.debug:
            # reduce data for test
            self.train_data = {
                i: ClientDataset(
                    self.dataset.trainset,
                    range(i * self.args.batch_size, (i + 1) * self.args.batch_size),
                )
                for i in range(self.args.num_clients)
            }
            self.test_data = {
                i: ClientDataset(
                    self.dataset.testset,
                    range(i * self.args.batch_size, (i + 1) * self.args.batch_size),
                )
                for i in range(self.args.num_clients)
            }

        else:
            self.train_data, self.test_data = self.sampler.sample(
                self.dataset.trainset, self.dataset.testset
            )

        # setup data for server
        self.train_data[-1] = None
        self.test_data[-1] = None

    def prepare(self) -> None:
        self.dataset = build_dataset(self.args.dataset)(self.args)
        self.sampler = build_sampler(self.args.sampler)(self.args)

        self.setup_data()

        self.client_list = [i for i in range(self.args.num_clients)]
        self.server = build_server(self.args.deploy_mode)(self.args)

    def run_round(self) -> None:
        pass

    def main(self) -> None:
        try:
            for round_idx in range(self.args.num_rounds):
                self.run_round(round_idx=round_idx)

            self.logger.info(f"all rounds finished.")

        except KeyboardInterrupt:
            self.server.model.save()
            self.logger.info(f"interrupted by user.")

    def evaluate_on_server(self) -> None:
        self.logger.info("evaluate on server")
        train_result = self.server.evaluate(
            data=self.train_data[-1], model=self.server.model
        )
        self.logger.info(f"train result: {train_result}")
        test_result = self.server.evaluate(
            data=self.test_data[-1], model=self.server.model
        )
        self.logger.info(f"test result: {test_result}")

        self.logger.info(
            f"final server model version: {self.server.model.get_model_version()}"
        )

    def evaluate_on_client(self) -> None:
        self.logger.info("evaluate on client")
        for client_id in self.client_list:
            client = build_client(self.args.deploy_mode)(
                self.args, client_id, data_size=len(self.train_data[client_id])
            )
            train_result = client.evaluate(
                data=self.train_data[client_id], model=self.server.model
            )
            self.logger.info(f"client {client_id} train result: {train_result}")
            test_result = client.evaluate(
                data=self.test_data[client_id], model=self.server.model
            )
            self.logger.info(f"client {client_id} test result: {test_result}")

        self.logger.info(
            f"final server model version: {self.server.model.get_model_version()}"
        )

    def finish(self) -> None:
        self.server.model.save()
        try:
            self.evaluate_on_server()
            if self.args.evaluate_on_client:
                self.evaluate_on_client()
        except KeyboardInterrupt:
            self.logger.info(f"interrupted by user.")
        except Exception as e:
            self.logger.error(f"error: {e}")

        self.logger.info(f"all finished.")

    def run(self) -> None:
        self.prepare()
        self.main()
        self.finish()
