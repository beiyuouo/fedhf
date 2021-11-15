#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   tests\test_component\test_selector.py
@Time    :   2021-11-09 19:55:55
@Author  :   Bingjie Yan
@Email   :   bj.yan.pa@qq.com
@License :   Apache License 2.0
"""

from fedhf.component import selector
from fedhf.component.selector import build_selector
from fedhf.api import opts


class TestSelector(object):
    client_list = [i for i in range(10)]
    args = opts().parse(['--num_clients', '10', '--select_ratio', '0.5'])

    def test_random_selector(self):
        selector = build_selector('random')(self.args)

        assert selector is not None
        assert selector.__class__.__name__ == 'RandomSelector'
        assert selector.select(self.client_list) is not None
        assert len(selector.select(self.client_list)) == int(
            self.args.num_clients * self.args.select_ratio)

    def test_random_async_selector(self):
        selector = build_selector('random_async')(self.args)

        assert selector is not None
        assert selector.__class__.__name__ == 'RandomAsyncSelector'
        assert selector.select(self.client_list) is not None

        selected = selector.select(self.client_list)
        assert len(selected) > 0
