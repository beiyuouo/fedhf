#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    :   fedhf\component\crypor\base_crypor.py
# @Time    :   2022-05-02 22:36:30
# @Author  :   Bingjie Yan
# @Email   :   bj.yan.pa@qq.com
# @License :   Apache License 2.0

from abc import ABC, abstractmethod

import numpy as np


class AbsEncryptor(ABC):
    """
    AbsEncryptor is the base class for all crypor mechanisms.
    """

    @abstractmethod
    def __init__(self, args, **kwargs):
        """
        Constructor.
        :param args: the arguments
        """
        self.args = args

    @abstractmethod
    def generate_noise(self):
        """
        Generate noise.
        :return: the noise
        """
        raise NotImplementedError

    @abstractmethod
    def clip_grad(self, model):
        """
        Clip the gradient of the given model.
        :param model: the model
        """
        raise NotImplementedError

    @abstractmethod
    def encrypt_model(self):
        """
        Encrypt the model.
        :return: the encrypted model
        """
        raise NotImplementedError


class BaseEncryptor(AbsEncryptor):
    """
    BaseEncryptor is the base class for all crypor mechanisms.
    """

    def __init__(self, args, **kwargs):
        """
        constructor.
        :param args: the arguments
        """
        super().__init__(args)

    def generate_noise(self, size):
        """
        generate noise.
        :param size: the size
        :return: the noise
        """
        return np.zeros(size)

    def clip_grad(self, model):
        """
        clip the gradient of the given model.
        :param model: the model
        """
        return

    def encrypt_model(self, model):
        """
        encrypt the given model.
        :param model: the model
        :return: the encrypted model
        """
        return model
