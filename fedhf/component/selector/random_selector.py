#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    :   fedhf\component\selector\random_selector.py
# @Time    :   2022-05-03 16:01:05
# @Author  :   Bingjie Yan
# @Email   :   bj.yan.pa@qq.com
# @License :   Apache License 2.0

import numpy as np

from .base_selector import BaseSelector


class RandomSelector(BaseSelector):
    def __init__(self, args) -> None:
        self.args = args

    def select(self, client_list: list) -> list:
        num_client_per_round = self.args.num_clients_per_round or int(self.args.num_clients * self.args.select_ratio)
        selected_clients = np.random.choice(client_list, num_client_per_round, replace=False)
        return selected_clients
