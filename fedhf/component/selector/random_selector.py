#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   fedhf\component\selector\random_selector.py
@Time    :   2021-11-08 21:16:49
@Author  :   Bingjie Yan
@Email   :   bj.yan.pa@qq.com
@License :   Apache License 2.0
"""

import numpy as np

from .base_selector import BaseSelector


class RandomSelector(BaseSelector):
    def __init__(self, args) -> None:
        self.args = args

    def select(self, client_list: list) -> list:
        selected_clients = np.random.choice(client_list, int(self.args.num_clients * self.args.select_ratio))
        return selected_clients
