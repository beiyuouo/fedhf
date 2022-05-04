#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    :   fedhf\core\injector\base_injector.py
# @Time    :   2022-05-03 16:02:52
# @Author  :   Bingjie Yan
# @Email   :   bj.yan.pa@qq.com
# @License :   Apache License 2.0

from fedhf.api import dp_mechanism_factory, dp_clip_factory
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
    'evaluator': evaluator_factory,
    'aggregator': aggregator_factory,
    'sampler': sampler_factory,
    'selector': selector_factory,
    'lr_schedule': lr_schedule_factory,
    'dp_mechanism': dp_mechanism_factory,
    'dp_clip': dp_clip_factory
}


class BaseInjector(object):

    def __init__(self):
        pass

    @classmethod
    def register(cls, component_name: str, component_dict: dict):
        if component_name not in components.keys():
            raise ValueError(f'Unknown component name: {component_name}')

        components[component_name].update(component_dict)