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
import numpy as np
import torch
import torch.distributed as dist
import pickle

from fedhf.api.cluster import Cluster
from fedhf.api.serial import Serializer, Deserializer


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
        self.cluster = Cluster(args)
        self.cluster.init()

    def send_tensor(self, tensor, dst, group=None):
        if group is None:
            pass

        dist.send(tensor.data, dst)

    def recv_tensor(self, tensor, src=None):
        recv = tensor.clone()
        dist.recv(recv.data, src=src)

    def send_obj(self, data, dst, group=None):
        if group is None:
            pass

        buf = pickle.dumps(data)
        size = torch.tensor(len(buf), dtype=torch.int32)
        arr = torch.from_numpy(np.copy(np.frombuffer(buf, dtype=np.int8)))

        resp0 = dist.isend(size, dst)
        resp1 = dist.isend(arr, dst)

        resp0.wait()
        resp1.wait()

    def recv_obj(self, src, group=None):
        if group is None:
            pass

        size = torch.tensor(1, dtype=torch.int32)
        dist.irecv(size, src=src, group=group).wait()

        data = torch.empty(size=(size, ), dtype=torch.int8)
        dist.irecv(data, src=src, group=group).wait()
        buf = data.numpy().tobytes()
        return Deserializer.restricted_loads(buf)