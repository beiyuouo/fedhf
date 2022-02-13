#!/usr/bin/env python
# -*- encoding: utf-8 -*-
""" 
@File    :   fedhf\core\server\distributed_server.py 
@Time    :   2021-12-06 13:53:21 
@Author  :   Bingjie Yan 
@Email   :   bj.yan.pa@qq.com 
@License :   Apache License 2.0 
"""
import queue
import threading
import torch

from fedhf.api import mc
from .base_server import BaseServer


class DistributedServer(BaseServer):
    """
    model in server is stored as a tensor
    """
    def __init__(self, args):
        super(DistributedServer, self).__init__(args)
        self.gradients_queue = queue.Queue()
        self.status = mc.SERVER_RUNNING_CODE

    def launch(self):
        # start a thread to agg gradients from clients
        self.thread_agg = threading.Thread(target=self.gradient_agg)
        self.thread_recv = threading.Thread(target=self.message_recv)
        self.thread_agg.start()
        self.thread_recv.start()

    def push_gradients(self, grad):
        self.gradients_queue.put(grad)

    def gradient_agg(self):
        while self.status != mc.SERVER_FINISHED_CODE:
            if self.gradients_queue.empty():
                continue
            grad = self.gradients_queue.get()
            result = self.aggregator.aggregate(grad)

            self.communicator.send(result, 0)  # fixme

    def message_recv(self):
        while self.status != mc.SERVER_FINISHED_CODE:
            msg = self.communicator.recv()
            if msg.message_code == mc.EXIT_CODE:
                self.close()
            elif msg.message_code == mc.REQUEST_CODE:
                self.communicator.send(self.model, 0)  # fixme
            elif msg.message_code == mc.SEND_CODE:
                self.push_gradients(msg.data)  # fixme
            else:
                raise ValueError("Unknown message code: {}".format(msg.message_code))

    def close(self):
        self.communicator.close()
        self.status = mc.FINISHED_CODE
        # self.thread_agg.join()
        # self.thread_recv.join()