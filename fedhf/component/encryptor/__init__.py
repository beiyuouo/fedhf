#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    :   fedhf\component\encryptor\__init__.py
# @Time    :   2022-05-02 22:57:05
# @Author  :   Bingjie Yan
# @Email   :   bj.yan.pa@qq.com
# @License :   Apache License 2.0

from .base_encryptor import AbsEncryptor, BaseEncryptor
from .dp_encryptor import DPEncryptor

encryptor_factory = {
    "none": BaseEncryptor,
    "base": BaseEncryptor,
    "dp": DPEncryptor,
}


def build_encryptor(encryptor_name):
    """
    Build the crypor mechanism.
    :param encryptor_name: the name of the crypor mechanism
    :return: the crypor mechanism
    """
    if encryptor_name not in encryptor_factory:
        raise ValueError("Unknown crypor mechanism: {}".format(encryptor_name))
    return encryptor_factory[encryptor_name]
