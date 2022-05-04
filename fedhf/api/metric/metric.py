#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    :   fedhf\api\metric\metric.py
# @Time    :   2022-03-19 17:50:33
# @Author  :   Bingjie Yan
# @Email   :   bj.yan.pa@qq.com
# @License :   Apache License 2.0

from typing import Dict, List
import json
import warnings


class Metric(object):

    def __init__(self, metric: Dict):
        self.metric = metric
        for _key in ['round', 'epoch', 'id']:
            if _key not in self.metric:
                self.metric[_key] = 0
                warnings.warn(f'Metric {_key} not found')

    def get_metric(self, name: str):
        if name in self.metric:
            return self.metric[name]
        raise KeyError(f'Metric {name} not found')

    def get_metrics(self) -> Dict:
        return self.metric

    def update(self, metric: Dict):
        self.metric.update(metric)

    def to_json(self) -> str:
        return json.dumps(self.metric)


class Metrics(object):

    def __init__(self, metrics: List[Metric]):
        self.metrics = metrics

    def add(self, metric: Metric):
        self.metrics.append(metric)

    def get_metrics(self) -> List[Metric]:
        return self.metrics

    def save(self, path: str):
        with open(path, 'w') as f:
            f.write(json.dumps(self.metrics))

    def load(self, path: str):
        with open(path, 'r') as f:
            self.metrics = json.loads(f.read())