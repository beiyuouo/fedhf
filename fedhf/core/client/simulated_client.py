#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    :   fedhf\core\client\simulated_client.py
# @Time    :   2022-03-19 22:07:10
# @Author  :   Bingjie Yan
# @Email   :   bj.yan.pa@qq.com
# @License :   Apache License 2.0

import re
from torch.utils.data import DataLoader

from fedhf.component import build_encryptor

from .base_client import BaseClient


class SimulatedClient(BaseClient):
    def __init__(self, args, client_id, **kwargs) -> None:
        super(SimulatedClient, self).__init__(args, client_id)
        assert "data_size" in kwargs.keys()
        self.data_size = kwargs["data_size"]

        if self.args.encryptor:
            self.encryptor = build_encryptor(self.args.encryptor)(self.args, data_size=self.data_size)

    def train(self, data, model, device="cpu", **kwargs):
        dataloader = DataLoader(data, batch_size=self.args.batch_size)

        result = self.trainer.train(
            dataloader=dataloader,
            model=model,
            num_epochs=self.args.num_local_epochs,
            client_id=self.client_id,
            device=device,
            encryptor=self.encryptor,
            **kwargs,
        )

        model = result["model"]
        result.pop("model")

        if self.args.encryptor:
            model = self.encryptor.encrypt_model(model)

        # self.logger.info(f'Finish training on client {self.client_id}, train_loss: {train_loss}')
        return model, result

    def evaluate(self, data, model, device="cpu", **kwargs):
        dataloader = DataLoader(data, batch_size=self.args.batch_size)

        result = self.evaluator.evaluate(dataloader=dataloader, model=model, client_id=self.client_id, device=device)
        if "model" in result:
            model = result["model"]
            result.pop("model")

        self.logger.info(f"Finish evaluating on client {self.client_id}, result: {result}")
        return result
