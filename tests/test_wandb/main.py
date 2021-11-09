#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   tests\test_wandb\main.py
@Time    :   2021-11-08 21:53:22
@Author  :   Bingjie Yan
@Email   :   bj.yan.pa@qq.com
@License :   Apache License 2.0
"""


import wandb


if __name__ == '__main__':
    wandb.init(project="my-test-project")
    train_acc = 0.3
    train_loss = 0.005
    wandb.log({'accuracy': train_acc, 'loss': train_loss})
    wandb.config.dropout = 0.2
