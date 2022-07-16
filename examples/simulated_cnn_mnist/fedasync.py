#!/usr/bin/env python
# -*- encoding: utf-8 -*-
""" 
@File    :   fedhf\examples\simulated_cnn_mnist\fedasync.py 
@Time    :   2021-11-22 19:58:55 
@Author  :   Bingjie Yan 
@Email   :   bj.yan.pa@qq.com 
@License :   Apache License 2.0 
"""

from fedhf.api import opts
from fedhf.core import SimulatedAsyncCoordinator

# '--use_wandb',


def main():
    args = opts().parse(
        [
            "--wandb_reinit",
            "--gpus",
            "0",
            "--deploy_mode",
            "simulated",
            "--scheme",
            "async",
            "--batch_size",
            "50",
            "--num_local_epochs",
            "3",
            "--resize",
            "--input_c",
            "1",
            "--image_size",
            "28",
            "--model",
            "cnn_mnist",
            "--dataset",
            "mnist",
            "--trainer",
            "fedasync_trainer",
            "--lr",
            "0.001",
            "--optim",
            "sgd",
            "--momentum",
            "0.9",
            "--num_clients",
            "100",
            "--num_rounds",
            "50",
            "--selector",
            "random_fedasync",
            "--select_ratio",
            "0.01",
            "--sampler",
            "non-iid",
            "--sampler_num_classes",
            "2",
            "--sampler_num_samples",
            "300",
            "--agg",
            "fedasync",
            "--fedasync_rho",
            "0.005",
            "--fedasync_strategy",
            "polynomial",
            "--fedasync_alpha",
            "0.9",
            "--fedasync_max_staleness",
            "4",
            "--fedasync_a",
            "0.5",
            "--fedasync_b",
            "4",
        ]
    )
    coo = SimulatedAsyncCoordinator(args)
    coo.run()


if __name__ == "__main__":
    main()
