#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    :   fedhf\component\crypor\base_crypor.py
# @Time    :   2022-05-02 22:36:30
# @Author  :   Bingjie Yan
# @Email   :   bj.yan.pa@qq.com
# @License :   Apache License 2.0

from abc import ABC, abstractmethod


class AbsEncryptor(ABC):
    """
    AbsEncryptor is the base class for all crypor mechanisms.
    """

    @abstractmethod
    def __init__(self, args):
        """
        Constructor.
        :param args: the arguments
        """
        self.args = args

    @abstractmethod
    def crypor(self, x):
        """
        Crypor the given data.
        :param x: the data
        :return: the crypor data
        """
        raise NotImplementedError


class BaseEncryptor(AbsEncryptor):
    """
    BaseEncryptor is the base class for all crypor mechanisms.
    """

    def __init__(self, args):
        """
        Constructor.
        :param args: the arguments
        """
        super().__init__(args)

    def encrypt_data(self, x):
        """
        Encrypt the given data.
        :param x: the data
        :return: the encrypted data
        """
        return x

    def encrypt_model(self, model):
        """
        Encrypt the given model.
        :param model: the model
        :return: the encrypted model
        """
        return model