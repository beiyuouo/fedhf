#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    :   fedhf\component\evaluator\evaluator.py
# @Time    :   2022-05-03 16:00:29
# @Author  :   Bingjie Yan
# @Email   :   bj.yan.pa@qq.com
# @License :   Apache License 2.0

from typing import Any, Optional, Union
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .base_evaluator import BaseEvaluator


class DefaultEvaluator(BaseEvaluator):
    def __init__(self, args) -> None:
        super(DefaultEvaluator, self).__init__(args)

    def evaluate(
        self,
        dataloader: DataLoader,
        model: nn.Module,
        client_id: Union[int, str] = None,
        gpus: list = [],
        device="cpu",
    ) -> Any:
        if len(gpus) > 1:
            pass
        else:
            pass
        if not client_id:
            client_id = -1
        model = model.to(device)
        crit = self.crit()

        self.logger.info(f"start evaluation on {client_id}")

        model.eval()
        losses = 0.0
        acc = 0.0
        for batch_idx, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = crit(outputs, labels)

            _, predicted = torch.max(outputs, 1)
            acc += torch.sum(predicted == labels).item()

            losses += loss.item()

        if dataloader and len(dataloader) > 0:
            losses /= len(dataloader)
            acc /= len(dataloader.dataset)

        self.logger.info(f"evaluation on {client_id}: loss {losses}, acc {acc}")
        self.logger.info(f"evaluation on {client_id} finished")

        if self.args.use_wandb:
            if client_id == -1:
                self.logger.to_wandb(
                    {
                        "acc on server": acc,
                        "loss on server": losses,
                        "epoch": model.get_model_version(),
                    }
                )

        return {"test_loss": losses, "test_acc": acc}
