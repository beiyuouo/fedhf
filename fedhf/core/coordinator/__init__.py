#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   fedhf\component\coordinator\__init__.py
@Time    :   2021-10-26 11:06:21
@Author  :   Bingjie Yan
@Email   :   bj.yan.pa@qq.com
@License :   Apache License 2.0
"""

__all__ = ["SimulatedCoordinator", "build_coordinator"]

from .simulated_coordinator import SimulatedCoordinator
from .simulated_async_coordinator import SimulatedAsyncCoordinator

coordinator_factory = {
    'simulated': SimulatedCoordinator,
    'simulated_async': SimulatedAsyncCoordinator,
}


def build_coordinator(coordinator_type):
    if coordinator_type not in coordinator_factory:
        raise ValueError(f'Unknown coordinator type: {coordinator_type}')
    coordinator = coordinator_factory[coordinator_type]
    return coordinator
