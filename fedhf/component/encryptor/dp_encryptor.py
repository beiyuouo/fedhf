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

    def __init__(self, args):
        """
        Constructor.
        :param args: the arguments
        """
        super().__init__(args)

    def generate_noise(self, size):
        """
        Encrypt the given data.
        :param x: the data
        :return: the encrypted data
        """
        return dpm.build_mechanism(
            self.args.dp_mechanism,
            dpm.calculate_sensitivity(self.args.lr, self.args.dp_clip, size),
            size,
            self.args.dp_epsilon,
            delta=self.args.dp_delta if hasattr(self.args, 'dp_delta') else None,
        )

    def clip_grad(self, model):
        """
        Clip the gradient of the given model.
        :param model: the model
        """
        return dpm.build_clip_grad(
            self.args.dp_mechanism,
            self.args.dp_clip,
            model,
        )

    def encrypt_model(self, model):
        """
        Encrypt the given model.
        :param model: the model
        :return: the encrypted model
        """
        model = self.clip_grad(model)

        with torch.no_grad():
            for k, v in model.named_parameters():
                noise = self.generate_noise(v.shape)
                noise = torch.from_numpy(noise).to(self.args.device)
                v.add_(noise)

        return model
