#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    :   fedhf\algor\async_\fedasync\fedasync_trainer.py
# @Time    :   2022-07-15 13:17:29
# @Author  :   Bingjie Yan
# @Email   :   bj.yan.pa@qq.com
# @License :   Apache License 2.0


import time
from copy import deepcopy

from tqdm import tqdm
from ....component.trainer.base_trainer import BaseTrainer


class FedAsyncTrainer(BaseTrainer):
    def __init__(self, args) -> None:
        super(FedAsyncTrainer, self).__init__(args)

        self.rho = args.fedasync.get("rho", 0.005)
        self.args.fedasync.update({"rho": self.rho})

    def train(
        self,
        dataloader,
        model,
        num_epochs,
        client_id=None,
        gpus=[],
        device="cpu",
        encryptor=None,
    ):
        if len(gpus) > 1:
            pass
        else:
            pass

        model_ = deepcopy(model)
        model = model.to(device)
        model_ = model_.to(device)
        if self.args.optim == "sgd":
            optim = self.optim(
                params=model.parameters(),
                lr=self.args.lr,
                momentum=self.args.momentum,
                weight_decay=self.args.weight_decay,
            )
        else:
            optim = self.optim(params=model.parameters(), lr=self.args.lr)
        crit = self.crit()
        lr_scheduler = self.lr_scheduler(optim, self.args.lr_step)

        self.logger.info(f"Start training on {client_id}")

        train_loss = []
        pbar = tqdm(total=num_epochs * len(dataloader))
        model.train()
        for epoch in range(num_epochs):
            losses = []
            pbar.set_description(
                f"Client:{client_id} Training on Epoch {epoch+1}/{num_epochs} Loss: {0 if len(train_loss)==0 else train_loss[-1]:.5f}"
            )
            for idx, (inputs, labels) in enumerate(dataloader):
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)

                # self.logger.info(f'labels: {labels}, outputs: {outputs}')

                l2_reg = self._calc_l2_reg(model_, model)

                loss = crit(outputs, labels) + l2_reg * self.args.fedasync_rho / 2

                # self.logger.info(
                #    f'ce: {crit(outputs, labels)}, l2_reg: {l2_reg}, loss: {loss.item()}')

                optim.zero_grad()

                if encryptor is not None:
                    encryptor.clip_grad(model)

                loss.backward()
                optim.step()

                losses.append(loss.item())
                pbar.update(1)

            train_loss.append(sum(losses) / len(losses))
            lr_scheduler.step()
            # self.logger.info(
            #    f'Client:{client_id} Epoch:{epoch+1}/{num_epochs} Loss:{train_loss[-1]}'
            # )

        time.sleep(0.3)
        self.logger.info(f"Client:{client_id} Train Loss:{train_loss}")

        if self.args.use_wandb and self.args.wandb_log_client:
            import wandb

            data = [[x, y] for (x, y) in zip(range(1, num_epochs + 1), train_loss)]
            table = wandb.Table(data=data, columns=["epoch", "train_loss"])
            self.logger.to_wandb(
                {
                    f"train at client {client_id} model_version {model.get_model_version()}": wandb.plot.line(
                        table,
                        "epoch",
                        "train_loss",
                        title=f"train loss at client {client_id}",
                    )
                }
            )

        return {"train_loss": train_loss, "model": model}

    def _calc_l2_reg(self, global_model, model):
        l2_reg = 0
        for p1, p2 in zip(global_model.parameters(), model.parameters()):
            l2_reg += (p1 - p2).norm(2)
        return l2_reg
