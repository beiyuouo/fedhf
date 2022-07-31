#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    :   fedhf\core\coordinator\simulated_sync_coordinator.py
# @Time    :   2022-05-03 16:02:43
# @Author  :   Bingjie Yan
# @Email   :   bj.yan.pa@qq.com
# @License :   Apache License 2.0

import os
from copy import deepcopy

from fedhf.core import build_client

from .base_coordinator import SimulatedBaseCoordinator


class SimulatedSyncCoordinator(SimulatedBaseCoordinator):
    """Simulated Coordinator
    In simulated scheme, the data and model belong to coordinator and there is no need communicator.
    Also, there is no need to instantiate every client.
    """

    def __init__(self, args) -> None:
        super(SimulatedSyncCoordinator, self).__init__(args)

    def prepare(self) -> None:
        super(SimulatedSyncCoordinator, self).prepare()

        assert self.args.deploy_mode == "simulated"

    def run_round(self, round_idx) -> None:
        selected_client = self.server.select(self.client_list)

        self.logger.info(f"round {round_idx} selected clients: {selected_client}")

        for client_id in selected_client:
            model = deepcopy(self.server.model)
            client = build_client(self.args.deploy_mode)(
                self.args, client_id, data_size=len(self.train_data[client_id])
            )
            model, result = client.train(self.train_data[client_id], model)
            self.server.update(
                model,
                server_model_version=self.server.model.get_model_version(),
                client_id=client_id,
                weight=len(self.train_data[client_id]),
            )

        if self.args.evaluate_on_client and round_idx % self.args.eval_interval == 0:
            self.evaluate_on_client()

        if self.server.model.get_model_version() % self.args.chkp_interval == 0:
            self.server.model.save(
                os.path.join(
                    self.args.save_dir,
                    f"{self.args.exp_name}-{self.server.model.get_model_version()}.pth",
                )
            )

    def finish(self) -> None:
        super(SimulatedSyncCoordinator, self).finish()

    def run(self) -> None:
        self.prepare()
        self.main()
        self.finish()
