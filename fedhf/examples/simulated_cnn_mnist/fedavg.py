#!/usr/bin/env python
# -*- encoding: utf-8 -*-
""" 
@File    :   fedhf\examples\simulated_cnn_mnist\fedavg.py 
@Time    :   2021-11-22 21:19:03 
@Author  :   Bingjie Yan 
@Email   :   bj.yan.pa@qq.com 
@License :   Apache License 2.0 
"""

from fedhf.api import opts
from fedhf.core import SimulatedSyncCoordinator

# '--use_wandb',


def main():
    args = opts().parse([
        '--wandb_reinit', '--gpus', '0', '--deploy_mode', 'simulated', '--scheme', 'sync',
        '--batch_size', '50', '--num_local_epochs', '5', '--resize', '--input_c', '3',
        '--image_size', '28', '--model', 'cnn_mnist', '--dataset', 'mnist', '--trainer',
        'trainer', '--lr', '0.01', '--optim', 'sgd', '--momentum', '0.9', '--num_clients', '10',
        '--num_rounds', '5', '--selector', 'random', '--select_ratio', '1', '--sampler',
        'non-iid', '--sampler_num_classes', '2', '--sampler_num_samples', '3000', '--agg',
        'fedavg'
    ])
    coo = SimulatedSyncCoordinator(args)
    coo.run()


if __name__ == "__main__":
    main()
