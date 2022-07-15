#!/usr/bin/env python 
# -*- coding: utf-8 -*- 
# @File    :   tests\test_dataset\test_clientdataset.py 
# @Time    :   2022-05-03 12:01:34 
# @Author  :   Bingjie Yan 
# @Email   :   bj.yan.pa@qq.com 
# @License :   Apache License 2.0 


from fedhf import Config
from fedhf.dataset import ClientDataset, build_dataset


class TestClientDataset(object):
    args = Config(num_classes=10, dataset="mnist")

    def test_clientdataset_mnist(self):
        dataset = build_dataset(self.args.dataset)(self.args)

        client_dataset = ClientDataset(dataset.trainset,
                                       [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

        assert len(client_dataset) == 10
        assert type(client_dataset) == ClientDataset
        assert type(client_dataset[0]) == tuple
        assert client_dataset.num_classes == 10
        assert client_dataset[0][0].equal(dataset.trainset[0][0])
        assert client_dataset[0][1] == dataset.trainset[0][1]
