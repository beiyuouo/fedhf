#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    :   fedhf\api\utils\system_utils.py
# @Time    :   2022-08-01 17:54:05
# @Author  :   Bingjie Yan
# @Email   :   bj.yan.pa@qq.com
# @License :   Apache License 2.0

import os
import torch


def get_system_flops(device="cpu"):
    """
    Get the FLOPS of the system.
    FLOPS: FLoating point Operations Per Second
    """
    raise NotImplementedError
    if device == "cpu":
        return 0
    elif device == "cuda" or device == "gpu":
        total_device_flops = 0
        for i in range(torch.cuda.device_count()):
            total_device_flops += torch.cuda.get_device_properties(i).total_compute_perf
        return total_device_flops
    else:
        raise ValueError("Unknown device: {}".format(device))
