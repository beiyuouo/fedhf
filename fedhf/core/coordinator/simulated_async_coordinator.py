#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    :   fedhf\core\coordinator\simulated_async_coordinator.py
# @Time    :   2022-05-03 16:02:37
# @Author  :   Bingjie Yan
# @Email   :   bj.yan.pa@qq.com
# @License :   Apache License 2.0

from copy import deepcopy
import os
import time
from typing import Any

import numpy as np

from fedhf.api import Config, utils, Serializer
from fedhf.core import build_client

from .base_coordinator import SimulatedBaseCoordinator


class SimulatedAsyncCoordinator(SimulatedBaseCoordinator):
    """Simulated Coordinator
    In simulated scheme, the data and model belong to coordinator and there is no need communicator.
    Also, there is no need to instantiate every client.
    """

    def __init__(self, args) -> None:
        super(SimulatedAsyncCoordinator, self).__init__(args)

    def prepare(self) -> None:
        super(SimulatedAsyncCoordinator, self).prepare()

        assert self.args.deploy_mode == "simulated"
        self._model_queue = []
        self._last_update_time = {client_id: 0 for client_id in self.client_list}
        self.max_staleness = self.args[self.args.algor].get("max_staleness", 16)
        self.server.model.save(os.path.join(self.args.weights_dir, "model-init.pth"))
        self._model_queue.append(os.path.join(self.args.weights_dir, "model-init.pth"))

    def run_round(self, round_idx) -> None:
        # self.logger.info(f'{self.server.model.get_model_version()}')
        selected_clients = self.server.select(self.client_list)
        self.logger.info(f"round {round_idx} selected clients: {selected_clients}")

        for client_id in selected_clients:
            client = build_client(self.args.deploy_mode)(
                self.args, client_id, data_size=len(self.train_data[client_id])
            )

            staleness = np.random.randint(
                low=1,
                high=min(
                    self.max_staleness,
                    max(self.server.model.get_model_version(), 0) + 1,
                    self.server.model.get_model_version()
                    - self._last_update_time[client_id]
                    + 1,
                )
                + 1,
            )

            assert staleness <= max(0, self.server.model.get_model_version()) + 1
            assert staleness <= len(self._model_queue)
            assert (
                staleness
                <= self.server.model.get_model_version()
                - self._last_update_time[client_id]
                + 1
            )

            self.logger.info(
                f"client {client_id} staleness: {staleness} load model from {self._model_queue[-staleness]}"
            )
            _model = deepcopy(self.server.model)
            _model = _model.load(self._model_queue[-staleness])

            self.logger.info(
                f"client {client_id} staleness: {staleness} start train from model version: {_model.get_model_version()}"
            )

            model, result = client.train(
                data=self.train_data[client_id],
                model=_model,
            )

            self.server.update(
                model,
                server_model_version=max(0, self.server.model.get_model_version()),
                client_model_version=max(0, model.get_model_version()),
            )

            if (
                self.args.evaluate_on_client
                and round_idx % self.args.eval_interval == 0
            ):
                self.evaluate_on_client()

            # save check point model
            if round_idx % self.args.chkp_interval == 0:
                self.logger.info(f"save model: {self.args.exp_name}-{round_idx}.pth")
                self.server.model.save(
                    os.path.join(
                        self.args.weights_dir,
                        f"{self.args.exp_name}-{round_idx}.pth",
                    )
                )

            # save temporary model
            self.logger.info(
                f"save temporary model: {self.args.exp_name}-model-tmp-{self.server.model.get_model_version()}.pth"
            )
            path = os.path.join(
                str(self.args.temp_dir),
                f"{self.args.exp_name}-model-tmp-{self.server.model.get_model_version()}.pth",
            )
            self.server.model.save(path)

            # append model path
            self._model_queue.append(path)
            # update time
            self._last_update_time[client_id] = self.server.model.get_model_version()

    def finish(self) -> None:
        super(SimulatedAsyncCoordinator, self).finish()

    def run(self) -> None:
        self.prepare()
        self.main()
        self.finish()


class SimulatedAsyncRandomCoordinator(SimulatedAsyncCoordinator):
    """Simulated Asynchorous Completely Random Coordinator"""

    def __init__(self, args) -> None:
        super(SimulatedAsyncRandomCoordinator, self).__init__(args)

    def prepare(self) -> None:
        super(SimulatedAsyncRandomCoordinator, self).prepare()
        # client start from 0
        self._client_update_queue = np.random.randint(
            self.args.num_clients, size=self.args.num_rounds
        )
        self.logger.info(f"client update queue: {self._client_update_queue}")

    def run_round(self, round_idx) -> None:
        # self.logger.info(f'{self.server.model.get_model_version()}')
        selected_clients = self.server.select(self.client_list)
        self.logger.info(f"round {round_idx} selected clients: {selected_clients}")

        for client_id in selected_clients:
            client = build_client(self.args.deploy_mode)(
                self.args, client_id, data_size=len(self.train_data[client_id])
            )

            staleness = (
                self.server.model.get_model_version()
                - self._last_update_time[client_id]
                + 1
            )

            _model = deepcopy(self.server.model)
            _model = _model.load(self._model_queue[-staleness])

            self.logger.info(
                f"client {client_id} staleness: {staleness} start train from model version: {_model.get_model_version()}"
            )

            model, result = client.train(
                data=self.train_data[client_id],
                model=_model,
            )

            self.server.update(
                model,
                server_model_version=max(0, self.server.model.get_model_version()),
                client_model_version=max(0, model.get_model_version()),
            )

            if (
                self.args.evaluate_on_client
                and round_idx % self.args.eval_interval == 0
            ):
                self.evaluate_on_client()

            # save check point model
            if round_idx % self.args.chkp_interval == 0:
                self.logger.info(f"save model: {self.args.exp_name}-{round_idx}.pth")
                self.server.model.save(
                    os.path.join(
                        self.args.weights_dir,
                        f"{self.args.exp_name}-{round_idx}.pth",
                    )
                )

            # save temporary model
            self.logger.info(
                f"save temporary model: {self.args.exp_name}-model-tmp-{self.server.model.get_model_version()}.pth"
            )
            path = os.path.join(
                self.args.temp_dir,
                f"{self.args.exp_name}-model-tmp-{self.server.model.get_model_version()}.pth",
            )
            self.server.model.save(path)

            # append model path
            self._model_queue.append(path)
            # update time
            self._last_update_time[client_id] = self.server.model.get_model_version()


class SimulatedAsyncEstimateCoordinator(SimulatedAsyncCoordinator):
    default_args = Config(
        simulated_async_real={"base_flops": 10, "base_bandwidth": 100}
    )
    # bandwidth: 100MBps

    def __init__(self, args) -> None:
        super(SimulatedAsyncEstimateCoordinator, self).__init__(args)

    def prepare(self) -> None:
        super(SimulatedAsyncEstimateCoordinator, self).prepare()
        # add default args to args
        self.add_default_args(self.default_args)

        self._client_update_queue = []
        self._client_time = np.zeros(self.args.num_clients)
        self._client_task_time = np.zeros(self.args.num_clients)

        # the FLOPS of each client is uniform the Gussian distribution with mean 0 and std 1
        self._client_flops = (
            np.random.randn(self.args.num_clients)
            + self.args.simulated_async_real.base_flops
        )
        # the bandwidth of each client is uniform the Gussian distribution with mean 0 and std 1
        self._client_bandwidth = (
            np.random.randn(self.args.num_clients)
            + self.args.simulated_async_real.base_bandwidth
        )
        self._client_train_time = np.zeros(self.args.num_clients)
        self._client_communication_time = np.zeros(self.args.num_clients)

        for client_id in range(self.args.num_clients):
            # init time is 0
            self._client_time[client_id] = 0

            # communication time is serialized model size / bandwidth
            self._client_communication_time[client_id] = (
                utils.get_communication_cost(  # Bytes
                    Serializer.serialize_model(self.server.model)
                )
                / 1024
                / 1024
                / (self._client_bandwidth[client_id] / 8)
            )

            # train time is FLOPS * model FLOPs * data_size / self.args.batch_size * epochs
            # imprecised, estimated by the model FLOPs
            self._client_train_time[client_id] = (
                utils.get_model_param_flops(
                    self.server.model,
                    (self.args.input_c, self.args.image_size, self.args.image_size),
                )[0]
                * self._client_flops[client_id]
                * (len(self.train_data[client_id]) // self.args.batch_size + 1)
                * self.args.num_epochs
            )

            # task time = communication time * 2 + calculation time
            self._client_task_time[client_id] = (
                self._client_train_time[client_id]
                + self._client_communication_time[client_id] * 2
            )

        # aggregation time estimation
        self._server_aggregated_time = (
            utils.get_model_param_flops(
                self.server.model,
                (self.args.input_c, self.args.image_size, self.args.image_size),
            )[
                0
            ]  # specific to the model
            * self.args.simulated_async_real.base_flops
        )
        self.global_time = 0

    def run_round(self, round_idx) -> None:
        # self.logger.info(f'{self.server.model.get_model_version()}')
        self._client_next_time = (
            self._client_time
            + self._client_train_time
            + self._client_communication_time
        )
        selected_clients = [np.argmin(self._client_next_time)]
        self.logger.info(f"round {round_idx} selected clients: {selected_clients}")

        for client_id in selected_clients:
            # add to queue
            self._client_update_queue.append(client_id)

            client = build_client(self.args.deploy_mode)(
                self.args, client_id, data_size=len(self.train_data[client_id])
            )

            staleness = (
                self.server.model.get_model_version()
                - self._last_update_time[client_id]
                + 1
            )

            _model = deepcopy(self.server.model)
            _model = _model.load(self._model_queue[-staleness])

            self.logger.info(
                f"client {client_id} staleness: {staleness} start train from model version: {_model.get_model_version()}"
            )

            model, result = client.train(
                data=self.train_data[client_id],
                model=_model,
            )

            # update model
            self.server.update(
                model,
                server_model_version=max(0, self.server.model.get_model_version()),
                client_model_version=max(0, model.get_model_version()),
            )

            # log time metrics
            # format: time, round, rank, event_group, event, value
            self.logger.log_metric(
                f"{time.time()}, {round_idx}, {client_id}, {'time'}, {'client_start_train'}, {self._client_time[client_id]}"
            )
            self.logger.log_metric(
                f"{time.time()}, {round_idx}, {client_id}, {'time'}, {'client_end_train'}, {self._client_time[client_id] + self._client_train_time[client_id]}"
            )
            self.logger.log_metric(
                f"{time.time()}, {round_idx}, {client_id}, {'time'}, {'client_start_communication'}, {self._client_time[client_id] + self._client_train_time[client_id]}"
            )
            self.logger.log_metric(
                f"{time.time()}, {round_idx}, {client_id}, {'time'}, {'client_end_communication'}, {self._client_time[client_id] + self._client_train_time[client_id] + self._client_communication_time[client_id]}"
            )

            # update time
            self.global_time = max(
                self.global_time,
                self._client_time[client_id]
                + self._client_train_time[client_id]
                + self._client_communication_time[client_id],
            )  # to make sure the global time is greater than the client time
            self.logger.log_metric(
                f"{time.time()}, {round_idx}, {-1}, {'time'}, {'server_start_aggregation'}, {self.global_time}"
            )

            self.global_time += self._server_aggregated_time  # add aggregation time
            self.logger.log_metric(
                f"{time.time()}, {round_idx}, {-1}, {'time'}, {'server_end_aggregation'}, {self.global_time}"
            )

            self._client_time[client_id] = (
                self.global_time + self._client_communication_time[client_id]
            )  # model send to client after aggregation
            self.logger.log_metric(
                f"{time.time()}, {round_idx}, {client_id}, {'time'}, {'client_start_communication'}, {self._client_time[client_id]}"
            )
            self.logger.log_metric(
                f"{time.time()}, {round_idx}, {client_id}, {'time'}, {'client_end_communication'}, {self._client_time[client_id] + self._client_communication_time[client_id]}"
            )

            if (
                self.args.evaluate_on_client
                and round_idx % self.args.eval_interval == 0
            ):
                self.evaluate_on_client()

            # save check point model
            if round_idx % self.args.chkp_interval == 0:
                self.logger.info(f"save model: {self.args.exp_name}-{round_idx}.pth")
                self.server.model.save(
                    os.path.join(
                        self.args.weights_dir,
                        f"{self.args.exp_name}-{round_idx}.pth",
                    )
                )

            # save temporary model
            self.logger.info(
                f"save temporary model: {self.args.exp_name}-model-tmp-{self.server.model.get_model_version()}.pth"
            )
            path = os.path.join(
                self.args.temp_dir,
                f"{self.args.exp_name}-model-tmp-{self.server.model.get_model_version()}.pth",
            )
            self.server.model.save(path)

            # append model path
            self._model_queue.append(path)
            # update time
            self._last_update_time[client_id] = self.server.model.get_model_version()
