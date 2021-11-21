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
    args = opts().parse([
        '--model', 'mlp', '--num_rounds', '3', '--num_local_epochs', '1', '--num_clients', '3',
        '--gpus', '-1', '--test'
    ])

    def test_simulated_async_coordinator(self):
        coordinator = SimulatedAsyncCoordinator(self.args)
        coordinator.run()

    def test_simulated_coordinator(self):
        self.args = opts().parse([
            '--model', 'mlp', '--num_rounds', '3', '--num_local_epochs', '1', '--num_clients',
            '3', '--gpus', '-1', '--test', '--agg', 'fedavg'
        ])

        coordinator = SimulatedCoordinator(self.args)
        coordinator.run()