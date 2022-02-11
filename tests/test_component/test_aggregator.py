#!/usr/bin/env python
# -*- encoding: utf-8 -*-
""" 
@File    :   tests\test_component\test_aggregator.py 
@Time    :   2021-11-11 13:48:17 
@Author  :   Bingjie Yan 
@Email   :   bj.yan.pa@qq.com 
@License :   Apache License 2.0 
"""

import torch
from fedhf import model
from fedhf.api import opts, Serializer
from fedhf.component import build_aggregator
from fedhf.model import build_model


class TestAggregator(object):
    args = opts().parse([
        '--num_classes', '10', '--model', 'mlp', '--gpus', '-1', '--agg', 'fedasync',
        '--fedasync_strategy', 'constant', '--fedasync_alpha', '0.5', '--select_ratio', '1'
    ])

    def test_fedasync_aggregator(self):
        agg = build_aggregator('fedasync')(self.args)

        model_1 = build_model(self.args.model)(self.args)
        model_2 = build_model(self.args.model)(self.args)

        model1_serialized = Serializer.serialize_model(model_1)
        model2_serialized = Serializer.serialize_model(model_2)

        result = agg.agg(model1_serialized,
                         model2_serialized,
                         server_model_version=2,
                         client_model_version=1)

        param = result['param']

        assert result['model_version'] == 3

    def test_fedavg_aggregator(self):
        self.args.num_clients = 2
        agg = build_aggregator('fedavg')(self.args)

        model = build_model(self.args.model)(self.args)

        model_param1 = Serializer.serialize_model(model)
        model_param2 = torch.zeros_like(model_param1)
        model_param3 = torch.ones_like(model_param1)

        result = agg.agg(model_param1,
                         model_param2,
                         client_id=0,
                         weight=1 / 2,
                         server_model_version=2)

        assert result is None

        result = agg.agg(model_param1,
                         model_param3,
                         client_id=1,
                         weight=1 / 2,
                         server_model_version=2)

        param = result['param']

        assert param[0] == 0.5 * model_param2[0] + 0.5 * model_param3[0]
        assert result['model_version'] == 3
