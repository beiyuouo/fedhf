#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    :   fedhf\core\coordinator\__init__.py
# @Time    :   2022-05-03 16:02:27
# @Author  :   Bingjie Yan
# @Email   :   bj.yan.pa@qq.com
# @License :   Apache License 2.0

__all__ = [
    "SimulatedSyncCoordinator", "SimulatedAsyncCoordinator", "build_coordinator",
    "coordinator_factory", "SimulatedBaseCoordinator"
]

from .base_coordinator import SimulatedBaseCoordinator

from .simulated_sync_coordinator import SimulatedSyncCoordinator
from .simulated_async_coordinator import SimulatedAsyncCoordinator

coordinator_factory = {
    'simulated_sync': SimulatedSyncCoordinator,
    'simulated_async': SimulatedAsyncCoordinator,
}


def build_coordinator(coordinator_type):
    if coordinator_type not in coordinator_factory:
        raise ValueError(f'Unknown coordinator type: {coordinator_type}')
    coordinator = coordinator_factory[coordinator_type]
    return coordinator
