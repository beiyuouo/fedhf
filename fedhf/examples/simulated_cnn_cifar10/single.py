#!/usr/bin/env python
# -*- encoding: utf-8 -*-
""" 
@File    :   fedhf\examples\simulated_cnn_cifar10\single.py 
@Time    :   2021-11-27 10:22:09 
@Author  :   Bingjie Yan 
@Email   :   bj.yan.pa@qq.com 
@License :   Apache License 2.0 
"""
from copy import deepcopy
import os
import torch

from fedhf.core import build_server, build_client
from fedhf.api import opts
from fedhf.component import Logger
from fedhf.dataset import ClientDataset, build_dataset
from fedhf.model import build_model

if __name__ == '__main__':
    args = opts().parse([
        '--wandb_reinit', '--gpus', '0', '--batch_size', '50', '--num_local_epochs', '100',
        '--resize', '--input_c', '3', '--image_size', '32', '--model', 'cnn_cifar10',
        '--dataset', 'cifar10', '--trainer', 'async_trainer', '--lr', '0.01', '--optim', 'sgd',
        '--momentum', '0.9', '--weight_decay', '0.00001', '--log_train_client'
    ])
    client = build_client('simulated')(args, client_id=0)
    dataset = build_dataset(args.dataset)(args)
    model = build_model(args.model)(args)

    if args.resume:
        model.load(os.path.join(args.save_dir, f'{args.name}-{model.get_model_version()}.pth'))

    model = client.train(dataset.trainset, model, args.device)
    model.save(os.path.join(args.save_dir, f'{args.name}-{model.get_model_version()}.pth'))
    result = client.evaluate(dataset.testset, model, args.device)