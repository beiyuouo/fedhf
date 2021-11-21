#!/usr/bin/env python
# -*- encoding: utf-8 -*-
""" 
@File    :   fedhf\examples\simulated_cnn_cifar10\fedavg.py 
@Time    :   2021-11-21 17:56:34 
@Author  :   Bingjie Yan 
@Email   :   bj.yan.pa@qq.com 
@License :   Apache License 2.0 
"""

from fedhf.api import opts
from fedhf.core import SimulatedCoordinator

# '--use_wandb',


def main():
    args = opts().parse([
        '--wandb_reinit', '--gpus', '0', '--batch_size', '50', '--num_local_epochs', '5',
        '--resize', '--input_c', '3', '--image_size', '24', '--model', 'cnn_cifar10',
        '--dataset', 'cifar10', '--trainer', 'trainer', '--lr', '0.01', '--optim', 'sgd',
        '--momentum', '0.75', '--num_clients', '10', '--num_rounds', '40', '--selector',
        'random', '--select_ratio', '1', '--sampler', 'random', '--sampler_num_classes', '2',
        '--sampler_num_samples', '250', '--agg', 'fedavg'
    ])
    coo = SimulatedCoordinator(args)
    coo.run()


if __name__ == "__main__":
    main()
