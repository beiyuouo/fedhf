#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    :   fedhf\api\metric\metric.py
# @Time    :   2022-03-19 17:50:33
# @Author  :   Bingjie Yan
# @Email   :   bj.yan.pa@qq.com
# @License :   Apache License 2.0

from typing import Dict


class Metric(object):

    def __init__(self, metrics: Dict):
        self.metrics = metrics

    def get_metrics(self) -> Dict:
        return self.metrics

    def update(self, metrics: Dict):
        self.metrics.update(metrics)
