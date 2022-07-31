#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    :   tests\test_component\test_aggregator.py
# @Time    :   2022-07-15 16:11:10
# @Author  :   Bingjie Yan
# @Email   :   bj.yan.pa@qq.com
# @License :   Apache License 2.0

import pytest
import torch
import fedhf
from fedhf.component import build_aggregator


@pytest.mark.order(3)
class TestAggregator(object):
    def test_agg_sync(self):
        args = fedhf.init(
            debug=True,
            num_clients=3,
            num_rounds=1,
            num_epochs=1,
            num_clients_per_round=3,
        )
        server_tensor = torch.Tensor([1, 1, 1, 1, 1])
        tensor_a = torch.Tensor([1, 3, 4, 5, 6])
        tensor_b = torch.Tensor([6, 5, 4, 3, 2])
        tensor_c = torch.Tensor([5, 4, 4, 7, 1])

        aggregator = build_aggregator("sync")(args)

        assert aggregator.__class__.__name__ == "SyncAggregator"
        result_a = aggregator.agg(server_param=server_tensor, client_param=tensor_a)
        result_b = aggregator.agg(server_param=server_tensor, client_param=tensor_b)
        result_c = aggregator.agg(server_param=server_tensor, client_param=tensor_c)

        assert result_a is None
        assert result_b is None
        assert result_c["param"].equal((tensor_a + tensor_b + tensor_c).div(3))

    def test_agg_async(self):
        args = fedhf.init(
            debug=True,
            num_clients=3,
            num_rounds=1,
            num_epochs=1,
            num_clients_per_round=1,
        )
        alpha = 0.6
        server_tensor = torch.Tensor([100, 100, 100, 100, 100])
        tensor_a = torch.Tensor([100, 300, 400, 500, 600])
        tensor_b = torch.Tensor([60, 50, 40, 30, 20])

        aggregator = build_aggregator("async")(args)

        assert aggregator.__class__.__name__ == "AsyncAggregator"
        result_a = aggregator.agg(server_param=server_tensor, client_param=tensor_a)
        result_b = aggregator.agg(server_param=result_a["param"], client_param=tensor_b)

        print(result_a["param"])
        print(result_b["param"])

        assert torch.all(
            result_a["param"].eq(server_tensor * (1 - alpha) + tensor_a * alpha)
        )
        assert torch.all(
            result_b["param"].eq(result_a["param"] * (1 - alpha) + tensor_b * alpha)
        )
