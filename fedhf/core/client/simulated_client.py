#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    :   fedhf\core\client\simulated_client.py
# @Time    :   2022-03-19 22:07:10
# @Author  :   Bingjie Yan
# @Email   :   bj.yan.pa@qq.com
# @License :   Apache License 2.0

import re
from torch.utils.data import DataLoader

from fedhf.component import Evaluator, build_trainer
from fedhf.model import build_criterion, build_model, build_optimizer

from .base_client import BaseClient


class SimulatedClient(BaseClient):

    def __init__(self, args, client_id) -> None:
        super(SimulatedClient, self).__init__(args, client_id)

    def train(self, data, model, device='cpu'):
        dataloader = DataLoader(data, batch_size=self.args.batch_size)

        result = self.trainer.train(dataloader=dataloader,
                                    model=model,
                                    num_epochs=self.args.num_local_epochs,
                                    client_id=self.client_id,
                                    device=device)
        # train_loss = result['train_loss']
        model = result['model']

        model = self.encryptor.encrypt_model(model)

        # self.logger.info(f'Finish training on client {self.client_id}, train_loss: {train_loss}')
        return model

    def evaluate(self, data, model, device='cpu'):
        dataloader = DataLoader(data, batch_size=self.args.batch_size)

        result = self.evaluator.evaluate(dataloader=dataloader,
                                         model=model,
                                         client_id=self.client_id,
                                         device=device)
        return result
        # self.logger.info(
        #     f'Finish evaluating on client {self.client_id}, test_loss: {result["test_loss"]} test_acc: {result["test_acc"]}'
        # )
