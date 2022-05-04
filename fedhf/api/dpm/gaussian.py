#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    :   fedhf\api\dpm\gaussian_noise.py
# @Time    :   2022-05-02 22:35:11
# @Author  :   Bingjie Yan
# @Email   :   bj.yan.pa@qq.com
# @License :   Apache License 2.0

import numpy as np
import torch


def gaussian_noise(sensitivity, size, epsilon, **kwargs):
    """
    Generate Gaussian noise with the given sensitivity.
    :param sensitivity: the sensitivity of the privacy mechanism
    :param size: the size of the noise
    :param epsilon: the privacy parameter
    :param kwargs: other parameters
    :return: the generated noise
    """
    assert 'delta' in kwargs, 'delta is required'
    delta = kwargs['delta']
    noise_scale = np.sqrt(2 * np.log(1.25 / delta)) * sensitivity / epsilon
    return np.random.normal(0, noise_scale, size)


def gaussian_clip(model: torch.nn.Module, clip: float):
    """
    Clip the model parameters.
    :param model: the model
    :param clip: the clipping bound
    :return: None
    """
    for k, v in model.named_parameters():
        v.grad /= max(1, v.grad.norm(2) / clip)