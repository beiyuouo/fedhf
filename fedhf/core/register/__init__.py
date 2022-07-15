#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    :   fedhf\core\register\__init__.py
# @Time    :   2022-07-15 13:05:22
# @Author  :   Bingjie Yan
# @Email   :   bj.yan.pa@qq.com
# @License :   Apache License 2.0


from fedhf.api import dp_mechanism_factory, dp_clip_factory
from fedhf.algor.base import algor_factory
from fedhf.component import (
    aggregator_factory,
    attactor_factory,
    encryptor_factory,
    evaluator_factory,
    sampler_factory,
    selector_factory,
    trainer_factory,
)
from fedhf.model import model_factory, optimizer_factory, criterion_factory, lr_schedule_factory
from fedhf.dataset import dataset_factory
from fedhf.core import coordinator_factory, server_factory, client_factory

components = {
    "dp_mechanism": dp_mechanism_factory,
    "dp_clip": dp_clip_factory,
    "algor": algor_factory,
    "aggregator": aggregator_factory,
    "agg": aggregator_factory,  # alias
    "attactor": attactor_factory,
    "encryptor": encryptor_factory,
    "evaluator": evaluator_factory,
    "sampler": sampler_factory,
    "selector": selector_factory,
    "trainer": trainer_factory,
    "model": model_factory,
    "optimizer": optimizer_factory,
    "criterion": criterion_factory,
    "loss": criterion_factory,  # alias
    "lr_schedule": lr_schedule_factory,
    "dataset": dataset_factory,
    "coordinator": coordinator_factory,
    "server": server_factory,
    "client": client_factory,
}


def register(component_name: str, component_dict: dict):
    if component_name not in components.keys():
        raise ValueError(f"Unknown component name: {component_name}")
    components[component_name].update(component_dict)


def register_all(component_dict: dict):
    for component_name, component_dict in component_dict.items():
        register(component_name, component_dict)
