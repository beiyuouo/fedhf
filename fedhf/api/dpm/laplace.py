#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    :   fedhf\api\dpm\laplace_noise.py
# @Time    :   2022-05-02 22:39:42
# @Author  :   Bingjie Yan
# @Email   :   bj.yan.pa@qq.com
# @License :   Apache License 2.0

import numpy as np
import torch


def laplace_noise(sensitivity, size, epsilon, **kwargs):
    """
    Generate Laplace noise with the given sensitivity.
    :param sensitivity: the sensitivity of the privacy mechanism
    :param size: the size of the noise
    :param epsilon: the privacy parameter
    :param kwargs: other parameters
    :return: the generated noise
    """
    noise_scale = sensitivity / epsilon
    return np.random.laplace(0, noise_scale, size)


def laplace_clip(model: torch.nn.Module, clip: float):
    """
    Clip the model parameters.
    :param model: the model
    :param clip: the clipping bound
    :return: None
    """
    for k, v in model.named_parameters():
        v.grad /= max(1, v.grad.norm(1) / clip)
