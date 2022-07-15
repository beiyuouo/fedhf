#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    :   fedhf\component\encryptor\dp_encryptor.py
# @Time    :   2022-05-02 22:55:08
# @Author  :   Bingjie Yan
# @Email   :   bj.yan.pa@qq.com
# @License :   Apache License 2.0

import torch
import torch.nn as nn

from .base_encryptor import BaseEncryptor

from fedhf.api import dpm


class DPEncryptor(BaseEncryptor):
    """
    DPEncryptor is the class for DP encryption.
    """

    def __init__(self, args, **kwargs):
        """
        Constructor.
        :param args: the arguments
        """
        super().__init__(args)
        assert kwargs["data_size"] is not None
        self.data_size = kwargs["data_size"]
        assert hasattr(args, "dp"), "The args must have the dp attribute."
        self.sensitivity = dpm.calculate_sensitivity(self.args.lr, self.args.dp.clip, self.data_size)

    def generate_noise(self, size):
        """
        Encrypt the given data.
        :param x: the data
        :return: the encrypted data
        """
        return dpm.build_mechanism(
            self.args.dp.mechanism,
            self.sensitivity,
            size,
            self.args.dp.epsilon,
            delta=self.args.dp.delta if hasattr(self.args.dp, "delta") else None,
        )

    def clip_grad(self, model):
        """
        Clip the gradient of the given model.
        :param model: the model
        """
        dpm.build_clip_grad(
            self.args.dp.mechanism,
            model,
            self.args.dp.clip,
        )

    def encrypt_model(self, model):
        """
        Encrypt the given model.
        :param model: the model
        :return: the encrypted model
        """
        self.clip_grad(model)

        with torch.no_grad():
            for k, v in model.named_parameters():
                noise = self.generate_noise(v.shape)
                noise = torch.from_numpy(noise).to(self.args.device)
                v.add_(noise)

        return model
