#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   fedhf\component\coordinator\single_coordinator.py
@Time    :   2021-10-26 11:06:00
@Author  :   Bingjie Yan
@Email   :   bj.yan.pa@qq.com
@License :   Apache License 2.0
"""


from base_coordinator import BaseCoordinator


class SingleCoordinator(BaseCoordinator):
    def __init__(self, args) -> None:
        self.args = args

    def init(self) -> None:
        # Build Server
        # Build Client
        # Build Aggregator
        # Build Model
        pass

    def main(self) -> None:
        pass

    def finish(self) -> None:
        pass

    def run(self) -> None:
        self.init()
        self.main()
        self.finish()
