#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    :   fedhf\core\coordinator\base_coordinator.py
# @Time    :   2022-05-03 15:41:08
# @Author  :   Bingjie Yan
# @Email   :   bj.yan.pa@qq.com
# @License :   Apache License 2.0

import time
import os
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any
from fedhf.api import Logger, Config
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
        self.dataset = build_dataset(self.args.dataset)(self.args)
        self.sampler = build_sampler(self.args.sampler)(self.args)

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

            for i in range(self.args.num_clients):
                self.logger.info(
                    f"client {i} train data: {len(self.train_data[i])} samples, test data: {len(self.test_data[i])} samples"
                )
            self.logger.info(
                f"server train data: {len(self.train_data[-1])} samples, test data: {len(self.test_data[-1])} samples"
            )

        # setup data for server
        if -1 not in self.train_data:
            self.train_data[-1] = None
        if -1 not in self.test_data:
            self.test_data[-1] = None

    def prepare(self) -> None:
        self.setup_data()

        self.client_list = [i for i in range(self.args.num_clients)]
        self.server = build_server(self.args.deploy_mode)(self.args)

    def run_round(self) -> None:
        pass

    def main(self) -> None:
        try:
            for round_idx in range(self.args.num_rounds):
                self.round_idx = round_idx
                self.run_round(round_idx=round_idx)

                self.check_eval(round_idx=round_idx)
                self.check_chkp(round_idx=round_idx)

            self.logger.info(f"all rounds finished.")

        except KeyboardInterrupt:
            self.server.model.save()
            self.logger.info(f"interrupted by user.")

    def log_metric(
        self, client_id: int = -1, metric: dict = None, event_type: str = "train"
    ) -> None:
        self.logger.info(f"client {client_id} {event_type} metric: {metric}")

        if metric is None:
            return

        for k, v in metric.items():
            # time, round, client, event_type, metric_name, metric_value
            self.logger.log_metric(
                f"{time.time()}, {self.round_idx}, {client_id}, {event_type}, {k}, {v}"
            )

    def evaluate_on_server(self) -> None:
        self.logger.info("evaluate on server")
        if -1 not in self.train_data and -1 not in self.test_data:
            self.logger.info("no train data on server")
            return
        if -1 in self.train_data:
            train_result = self.server.evaluate(
                data=self.train_data[-1], model=self.server.model
            )
            self.logger.info(f"train result: {train_result}")

        if -1 in self.test_data:
            test_result = self.server.evaluate(
                data=self.test_data[-1], model=self.server.model
            )
            self.logger.info(f"test result: {test_result}")

        self.logger.info(
            f"final server model version: {self.server.model.get_model_version()}"
        )
        self.log_metric(-1, train_result, "train")
        self.log_metric(-1, test_result, "test")

    def evaluate_on_client(self) -> None:
        self.logger.info("evaluate on client")
        total_train_result = {}
        total_test_result = {}
        for client_id in self.client_list:
            client = build_client(self.args.deploy_mode)(
                self.args, client_id, data_size=len(self.train_data[client_id])
            )
            train_result = client.evaluate(
                data=self.train_data[client_id], model=self.server.model
            )
            self.logger.info(f"client {client_id} train result: {train_result}")
            train_result.update({"data_size": len(self.train_data[client_id])})
            total_train_result[client_id] = train_result

            test_result = client.evaluate(
                data=self.test_data[client_id], model=self.server.model
            )
            self.logger.info(f"client {client_id} test result: {test_result}")
            test_result.update({"data_size": len(self.test_data[client_id])})
            total_test_result[client_id] = test_result

            self.log_metric(client_id, train_result, "train")
            self.log_metric(client_id, test_result, "test")

        if len(self.client_list) <= 0:
            self.logger.info(f"no client to evaluate")
            return

        total_train_metric = {}
        total_test_metric = {}
        for client_id in self.client_list:
            for key in total_train_result[client_id]:
                if key not in total_train_metric:
                    total_train_metric[key] = 0.0

                if key != "data_size":
                    total_train_metric[key] += (
                        total_train_result[client_id][key]
                        * total_train_result[client_id]["data_size"]
                    )
                else:
                    total_train_metric[key] += total_train_result[client_id][key]

            for key in total_test_result[client_id]:
                if key not in total_test_metric:
                    total_test_metric[key] = 0.0

                if key != "data_size":
                    total_test_metric[key] += (
                        total_test_result[client_id][key]
                        * total_test_result[client_id]["data_size"]
                    )
                else:
                    total_test_metric[key] += total_test_result[client_id][key]

        assert total_train_metric["data_size"] > 0, "data_size is zero"

        for key in total_train_metric:
            if key != "data_size" and total_train_metric["data_size"] > 0:
                total_train_metric[key] /= total_train_metric["data_size"]

        for key in total_test_metric:
            if key != "data_size" and total_test_metric["data_size"] > 0:
                total_test_metric[key] /= total_test_metric["data_size"]

        self.logger.info(f"total train result: {total_train_metric}")
        self.logger.info(f"total test result: {total_test_metric}")

        self.log_metric(-1, total_train_metric, "train")
        self.log_metric(-1, total_test_metric, "test")

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

    def add_default_args(self, args=None) -> Any:
        if args is None:
            if not hasattr(self, "default_args"):
                args = Config()
            else:
                args = deepcopy(self.default_args)
        # print("func args:", args)
        self.args.merge(args, overwrite=False)

    def check_eval(self, round_idx):
        if (round_idx + 1) % self.args.eval_interval == 0:
            if self.args.evaluate_on_client:
                self.evaluate_on_client()
            self.evaluate_on_server()

    def check_chkp(self, round_idx):
        if (round_idx + 1) % self.args.chkp_interval == 0:
            self.server.model.save(
                os.path.join(
                    self.args.save_dir,
                    f"{self.args.exp_name}-{self.server.model.get_model_version()}.pth",
                )
            )
