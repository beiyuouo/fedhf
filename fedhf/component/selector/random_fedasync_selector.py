#!/usr/bin/env python
# -*- encoding: utf-8 -*-
""" 
@File    :   fedhf\component\selector\random_async_selector.py 
@Time    :   2021-11-15 23:02:09 
@Author  :   Bingjie Yan 
@Email   :   bj.yan.pa@qq.com 
@License :   Apache License 2.0 
"""

import numpy as np

from .base_selector import BaseSelector


class RandomFedAsyncSelector(BaseSelector):
    def __init__(self, args) -> None:
        self.args = args

    def select(self, client_list: list) -> list:
        # select_ratio = np.random.rand()
        select_ratio = self.args.select_ratio
        select_count = min(int(self.args.num_clients * select_ratio),
                           self.args.fedasync_max_staleness)
        select_count = max(1, select_count)
        selected_clients = np.random.choice(client_list, select_count, replace=False)
        # np.random.shuffle(selected_clients)
        return selected_clients
