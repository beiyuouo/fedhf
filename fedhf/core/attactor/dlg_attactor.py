#!/usr/bin/env python
# -*- encoding: utf-8 -*-
""" 
@File    :   fedhf\core\attactor\dlg_attactor.py 
@Time    :   2021-12-13 16:18:04 
@Author  :   Bingjie Yan 
@Email   :   bj.yan.pa@qq.com 
@License :   Apache License 2.0 
"""

from .base_attactor import BaseAttactor


class DLGAttactor(BaseAttactor):
    def __init__(self, args):
        super(DLGAttactor, self).__init__(args)