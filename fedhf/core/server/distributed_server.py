#!/usr/bin/env python
# -*- encoding: utf-8 -*-
""" 
@File    :   fedhf\core\server\distributed_server.py 
@Time    :   2021-12-06 13:53:21 
@Author  :   Bingjie Yan 
@Email   :   bj.yan.pa@qq.com 
@License :   Apache License 2.0 
"""

import torch
from .base_server import BaseServer


class DistributedServer(BaseServer):
    def __init__(self, args):
        super(DistributedServer, self).__init__(args)

    def launch(self):
        pass