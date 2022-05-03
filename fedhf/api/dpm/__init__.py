#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    :   fedhf\api\dpm\__init__.py
# @Time    :   2022-05-02 22:15:36
# @Author  :   Bingjie Yan
# @Email   :   bj.yan.pa@qq.com
# @License :   Apache License 2.0

# dpm is a package for differential privacy mechanisms

import numpy as np
from .sensitivity import calculate_sensitivity
from .gaussian import gaussian_noise, gaussian_clip
from .laplace import laplace_noise, laplace_clip

mechanism_factory = {
    'none': None,
    'gaussian': gaussian_noise,
    'laplace': laplace_noise,
}

clip_grad_factory = {
    'none': None,
    'gaussian': gaussian_clip,
    'laplace': laplace_clip,
}


def build_mechanism(mechanism, sensitivity, size, epsilon, **kwargs):
    """
    Build the privacy mechanism.
    :param mechanism: the mechanism name
    :param sensitivity: the sensitivity of the privacy mechanism
    :param size: the size of the noise
    :param epsilon: the privacy parameter
    :param kwargs: other parameters
    :return: the privacy mechanism
    """
    if mechanism == 'none':
        return np.zeros(size)
    elif mechanism in mechanism_factory:
        return mechanism_factory[mechanism](sensitivity, size, epsilon, **kwargs)
    else:
        raise ValueError('Unknown mechanism: %s' % mechanism)


def build_clip_grad(mechanism, model, clip, **kwargs):
    """
    Clip the gradients of the model.
    :param mechanism: the mechanism name
    :param model: the model
    :param clip: the clipping bound
    :param kwargs: other parameters
    :return: None
    """
    if mechanism == 'none':
        return
    elif mechanism in clip_grad_factory:
        clip_grad_factory[mechanism](model, clip, **kwargs)
    else:
        raise ValueError('Unknown mechanism: %s' % mechanism)


__all__ = [
    'calculate_sensitivity',
    'gaussian_noise',
    'gaussian_clip',
    'laplace_noise',
    'laplace_clip',
    'build_mechanism',
    'build_clip_grad',
]
