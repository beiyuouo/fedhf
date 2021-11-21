#!/usr/bin/env python
# -*- encoding: utf-8 -*-
""" 
@File    :   tests\test_core\test_coordinator.py 
@Time    :   2021-11-15 19:59:12 
@Author  :   Bingjie Yan 
@Email   :   bj.yan.pa@qq.com 
@License :   Apache License 2.0 
"""

from fedhf.api import opts
from fedhf.core import SimulatedCoordinator, SimulatedAsyncCoordinator


class TestCoordinator(object):
    def test_simulated_async_coordinator(self):
        args = opts().parse([
            '--model', 'mlp', '--num_rounds', '3', '--num_local_epochs', '1', '--num_clients',
            '3', '--gpus', '-1', '--test', '--select_ratio', '0.5', '--agg', 'fedasync'
        ])

        coordinator = SimulatedAsyncCoordinator(args)
        coordinator.run()

    def test_simulated_coordinator(self):
        args = opts().parse([
            '--model', 'mlp', '--num_rounds', '3', '--num_local_epochs', '1', '--num_clients',
            '3', '--gpus', '-1', '--test', '--agg', 'fedavg', '--select_ratio', '0.5'
        ])

        coordinator = SimulatedCoordinator(args)
        coordinator.run()