#!/usr/bin/env python
# -*- encoding: utf-8 -*-
""" 
@File    :   fedhf\component\communicator\distributed_communicator.py 
@Time    :   2022-01-27 22:59:55 
@Author  :   Bingjie Yan 
@Email   :   bj.yan.pa@qq.com 
@License :   Apache License 2.0 
"""

from .base_communicator import BaseCommunicator


class DistributedCommunicator(BaseCommunicator):
    def __init__(self, args):
        super(DistributedCommunicator, self).__init__(args)
