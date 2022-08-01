#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    :   fedhf\core\coordinator\simulated_async_coordinator.py
# @Time    :   2022-05-03 16:02:37
# @Author  :   Bingjie Yan
# @Email   :   bj.yan.pa@qq.com
# @License :   Apache License 2.0

from copy import deepcopy
import os
from typing import Any

import numpy as np

from fedhf.api import Config
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


class SimulatedAsyncRealCoordinator(SimulatedAsyncCoordinator):
    default_args = Config(
        simulated_async_real={"base_flops": 100, "base_bandwidth": 100}
    )

    def __init__(self, args) -> None:
        super(SimulatedAsyncRealCoordinator, self).__init__(args)

    def prepare(self) -> None:
        self._client_update_queue = []
        self.client_time = []
        # the FLOPS of each client is uniform the Gussian distribution with mean 0 and std 1
        self.client_flops = (
            np.random.randn(self.args.num_clients)
            + self.args.simulated_async_real.base_flops
        )
        # the bandwidth of each client is uniform the Gussian distribution with mean 0 and std 1
        self.client_bandwidth = (
            np.random.randn(self.args.num_clients)
            + self.args.simulated_async_real.base_bandwidth
        )
