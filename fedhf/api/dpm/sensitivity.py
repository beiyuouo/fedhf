#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    :   fedhf\api\dpm\sensitivity.py
# @Time    :   2022-05-02 22:19:28
# @Author  :   Bingjie Yan
# @Email   :   bj.yan.pa@qq.com
# @License :   Apache License 2.0

import numpy as np


def calculate_sensitivity(lr, clip, data_size):
    """
    Calculate the sensitivity of the privacy mechanism.
    :param lr: learning rate
    :param clip: clipping bound
    :param data_size: data size
    :return: the sensitivity of the privacy mechanism
    """
    return 2 * lr * clip / data_size