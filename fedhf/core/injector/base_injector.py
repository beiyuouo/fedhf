#!/usr/bin/env python
# -*- encoding: utf-8 -*-
""" 
@File    :   fedhf\api\injector\base_injector.py 
@Time    :   2021-11-30 22:54:03 
@Author  :   Bingjie Yan 
@Email   :   bj.yan.pa@qq.com 
@License :   Apache License 2.0 
"""

from fedhf.component import aggregator_factory, trainer_factory, evaluator_factory, sampler_factory, selector_factory
from fedhf.model import model_factory, optimizer_factory, criterion_factory, lr_schedule_factory
from fedhf.dataset import dataset_factory
from fedhf.core import coordinator_factory, server_factory, client_factory

components = {
    'coordinator': coordinator_factory,
    'server': server_factory,
    'client': client_factory,
    'model': model_factory,
    'optimizer': optimizer_factory,
    'criterion': criterion_factory,
    'dataset': dataset_factory,
    'trainer': trainer_factory,
    'evalutor': evaluator_factory,
    'aggregator': aggregator_factory,
    'sampler': sampler_factory,
    'selector': selector_factory,
    'lr_schedule': lr_schedule_factory
}


class BaseInjector(object):
    def __init__(self):
        pass

    @classmethod
    def register(cls, component_name: str, component_dict: dict):
        if component_name not in components.keys():
            raise ValueError(f'Unknown component name: {component_name}')

        components[component_name].update(component_dict)