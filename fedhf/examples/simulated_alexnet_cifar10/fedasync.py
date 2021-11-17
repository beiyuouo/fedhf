#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   fedhf\examples\simulated-cnn-mnist\fedavg.py
@Time    :   2021-11-08 21:44:45
@Author  :   Bingjie Yan
@Email   :   bj.yan.pa@qq.com
@License :   Apache License 2.0
"""

from fedhf.api import opts
from fedhf.core import SimulatedAsyncCoordinator


def main():
    args = opts().parse([
        '--use_wandb', '--wandb_reinit', '--gpus', '0', '--batch_size', '50',
        '--num_local_epochs', '5', '--resize', False, '--model',
        'alexnet_cifar10', '--dataset', 'cifar10', '--lr', '0.1', '--optim',
        'sgd', '--num_clients', '100', '--num_rounds', '13', '--fedasync_rho',
        '0.005', '--fedasync_strategy', 'constant', '--fedasync_alpha', '0.5',
        '--fedasync_max_staleness', '4', '--fedasync_a', '10', '--fedasync_b',
        '4'
    ])
    coo = SimulatedAsyncCoordinator(args)
    coo.run()


if __name__ == "__main__":
    main()
