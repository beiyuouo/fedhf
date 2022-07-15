#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    :   fedhf\core\coordinator\simulated_sync_coordinator.py
# @Time    :   2022-05-03 16:02:43
# @Author  :   Bingjie Yan
# @Email   :   bj.yan.pa@qq.com
# @License :   Apache License 2.0

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

    def main(self) -> None:
        try:
            for i in range(self.args.num_rounds):
                selected_client = self.server.select(self.client_list)

                self.logger.info(f"Round {i} selected client: {selected_client}")

                for client_id in selected_client:
                    model = deepcopy(self.server.model)
                    client = build_client(self.args.deploy_mode)(
                        self.args, client_id, data_size=len(self.data[client_id])
                    )
                    model = client.train(self.data[client_id], model)
                    self.server.update(
                        model, server_model_version=self.server.model.get_model_version(), client_id=client_id
                    )

                result = self.server.evaluate(self.dataset.testset)
                self.logger.info(f"Server result: {result}")

                if self.server.model.get_model_version() % self.args.checkpoint_interval == 0:
                    self.server.model.save(f"{self.args.name}-{self.server.model.get_model_version()}.pth")

            self.logger.info(f"All rounds finished.")

        except KeyboardInterrupt:
            self.server.model.save()
            self.logger.info(f"Interrupted by user.")

    def finish(self) -> None:
        super(SimulatedSyncCoordinator, self).finish()

    def run(self) -> None:
        self.prepare()
        self.main()
        self.finish()
