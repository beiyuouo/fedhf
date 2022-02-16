#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    :   examples\distributed_cnn_mnist\run.py
# @Time    :   2022-02-15 15:10:25
# @Author  :   Bingjie Yan
# @Email   :   bj.yan.pa@qq.com
# @License :   Apache License 2.0

from fedhf.api import opts
from fedhf.core import *

args = opts().parse([
    '--wandb_reinit', '--gpus', '0', '--deploy_mode', 'distributed', '--scheme', 'async',
    '--batch_size', '50', '--num_local_epochs', '3', '--resize', '--input_c', '1',
    '--image_size', '28', '--model', 'cnn_mnist', '--dataset', 'mnist', '--trainer',
    'fedasync_trainer', '--lr', '0.001', '--optim', 'sgd', '--momentum', '0.9', '--num_clients',
    '100', '--num_rounds', '50', '--selector', 'random_fedasync', '--select_ratio', '0.01',
    '--sampler', 'non-iid', '--sampler_num_classes', '2', '--sampler_num_samples', '300',
    '--agg', 'fedasync', '--fedasync_rho', '0.005', '--fedasync_strategy', 'polynomial',
    '--fedasync_alpha', '0.9', '--fedasync_max_staleness', '4', '--fedasync_a', '0.5',
    '--fedasync_b', '4'
])


def launch_coordinator(args, ip, port, rank):
    args.addr = f'{ip}'
    args.port = f'{port}'
    args.rank = f'{rank}'
    args.world_size = f'{args.num_clients+2}'

    coor = build_coordinator('distributed')(args)
    coor.launch()


def launch_server(ip, port, rank):
    pass


def launch_client(ip, port, rank):
    pass


if __name__ == '__main__':
    launch_coordinator()
    launch_server()
    launch_client()