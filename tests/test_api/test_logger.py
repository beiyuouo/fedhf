#!/usr/bin/env python
# -*- encoding: utf-8 -*-
""" 
@File    :   tests\test_component\test_logger.py 
@Time    :   2021-11-11 13:19:25 
@Author  :   Bingjie Yan 
@Email   :   bj.yan.pa@qq.com 
@License :   Apache License 2.0 
"""

from fedhf.api import opts, Logger


class TestLogger:
    args = opts().parse(['--project_name', 'fedhf'])

    def test_logger(self):
        self.logger = Logger(self.args)