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

    @classmethod
    def interrupt_exception(self, func):
        def wrapper(*args, **kwargs):
            try:
                func(*args, **kwargs)
            except KeyboardInterrupt:
                self.server.model.save()
                self.logger.info(f"interrupted by user.")

        return wrapper

    def prepare(self) -> None:
        self.dataset = build_dataset(self.args.dataset)(self.args)
        self.sampler = build_sampler(self.args.sampler)(self.args)

        if self.args.debug:
            # reduce data for test
            self.data = [
                ClientDataset(
                    self.dataset.trainset,
                    range(i * self.args.batch_size, (i + 1) * self.args.batch_size),
                )
                for i in range(self.args.num_clients)
            ]
        else:
            self.data = self.sampler.sample(self.dataset.trainset)

        self.client_list = [i for i in range(self.args.num_clients)]
        self.server = build_server(self.args.deploy_mode)(self.args)

    def main(self) -> None:
        pass

    def evaluate_on_server(self) -> None:
        pass

    def evaluate_on_client(self) -> None:
        self.logger.info("evaluate on client")
        for client_id in self.client_list:
            client = build_client(self.args.deploy_mode)(
                self.args, client_id, data_size=len(self.data[client_id])
            )
            result = client.evaluate(data=self.data[client_id], model=self.server.model)
            self.logger.info(f"client {client_id} result: {result}")
        result = self.server.evaluate(self.dataset.testset)
        self.logger.info(f"server result: {result}")
        self.logger.info(
            f"final server model version: {self.server.model.get_model_version()}"
        )

    def finish(self) -> None:
        self.server.model.save()
        if self.args.evaluate_on_client:
            try:
                self.evaluate_on_client()
            except KeyboardInterrupt:
                self.logger.info(f"interrupted by user.")
        self.logger.info(f"All finished.")

    def run(self) -> None:
        self.prepare()
        self.main()
        self.finish()
