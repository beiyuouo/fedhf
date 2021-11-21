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

# '--use_wandb',


def main():
    args = opts().parse([
        '--wandb_reinit', '--gpus', '0', '--batch_size', '50', '--num_local_epochs', '5',
        '--resize', '--input_c', '3', '--image_size', '24', '--model', 'cnn_cifar10',
        '--dataset', 'cifar10', '--trainer', 'async_trainer', '--lr', '0.01', '--optim', 'sgd',
        '--momentum', '0.75', '--num_clients', '100', '--num_rounds', '400', '--selector',
        'random_async', '--select_ratio', '0.01', '--sampler', 'non-iid',
        '--sampler_num_classes', '2', '--sampler_num_samples', '250', '--fedasync_rho', '0.005',
        '--fedasync_strategy', 'polynomial', '--fedasync_alpha', '0.9',
        '--fedasync_max_staleness', '4', '--fedasync_a', '0.5', '--fedasync_b', '4'
    ])
    coo = SimulatedAsyncCoordinator(args)
    coo.run()


if __name__ == "__main__":
    main()
