#!/usr/bin/env python
# -*- encoding: utf-8 -*-
""" 
@File    :   fedhf\component\communicator\base_communicator.py 
@Time    :   2021-12-06 16:16:44 
@Author  :   Bingjie Yan 
@Email   :   bj.yan.pa@qq.com 
@License :   Apache License 2.0 
"""

from abc import ABC, abstractmethod
import torch
import torch.distributed as dist

from fedhf.api.network.network import Network


class AbsCommunicator(ABC):
    @abstractmethod
    def send(self, msg):
        raise NotImplementedError

    @abstractmethod
    def recv(self):
        raise NotImplementedError


class BaseCommunicator(AbsCommunicator):
    """
    Base class for communicator.
    """
    def __init__(self, args):
        self.network = Network(args)
        self.network.init()

    def send(self, data):
        pass

    def recv(self):
        pass