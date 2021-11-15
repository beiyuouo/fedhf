#!/usr/bin/env python
# -*- encoding: utf-8 -*-
""" 
@File    :   tests\test_component\test_aggregator.py 
@Time    :   2021-11-11 13:48:17 
@Author  :   Bingjie Yan 
@Email   :   bj.yan.pa@qq.com 
@License :   Apache License 2.0 
"""

from fedhf import model
from fedhf.api import opts
from fedhf.component import Serializer, build_aggregator
from fedhf.model import build_model


class TestAggregator(object):
    args = opts().parse([
        '--num_classes', '10', '--model', 'mlp', '--gpus', '-1', '--agg',
        'fedasync', '--fedasync_strategy', 'constant', '--fedasync_alpha',
        '0.5'
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
