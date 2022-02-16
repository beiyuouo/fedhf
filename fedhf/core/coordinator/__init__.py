#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   fedhf\component\coordinator\__init__.py
@Time    :   2021-10-26 11:06:21
@Author  :   Bingjie Yan
@Email   :   bj.yan.pa@qq.com
@License :   Apache License 2.0
"""

__all__ = [
    "SimulatedSyncCoordinator", "SimulatedAsyncCoordinator", "build_coordinator",
    "coordinator_factory", "DistributedCoordinator"
]

from .simulated_sync_coordinator import SimulatedSyncCoordinator
from .simulated_async_coordinator import SimulatedAsyncCoordinator
from .distributed_coordinator import DistributedCoordinator

coordinator_factory = {
    'simulated_sync': SimulatedSyncCoordinator,
    'simulated_async': SimulatedAsyncCoordinator,
    'distributed': DistributedCoordinator
}


def build_coordinator(coordinator_type):
    if coordinator_type not in coordinator_factory:
        raise ValueError(f'Unknown coordinator type: {coordinator_type}')
    coordinator = coordinator_factory[coordinator_type]
    return coordinator
