#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    :   fedhf\component\sampler\base_sampler.py
# @Time    :   2022-05-03 16:00:39
# @Author  :   Bingjie Yan
# @Email   :   bj.yan.pa@qq.com
# @License :   Apache License 2.0

from abc import ABC
from copy import deepcopy
from typing import Any, List
import torch.utils.data as data

from fedhf.api import Config
from fedhf.api.utils.json_utils import NpEncoder


class BaseSampler(ABC):
    def __init__(self, args) -> None:
        self.args = args

    def sample(self):
        raise NotImplementedError

    def split_dataset(self, data_idxs: List, test_size: int = 0.2):
        """
        split dataset into train and test
        """
        from sklearn.model_selection import train_test_split

        train_data, test_data = train_test_split(data_idxs, test_size=test_size)
        return train_data, test_data

    def export_data_partition(self, train_data, test_data):
        # export to json
        import json

        json_path = self.args.save_dir / "train_data.json"

        if type(train_data) is dict:
            with json_path.open("w") as f:
                json.dump(train_data, f, cls=NpEncoder)
        elif type(train_data) is list:

            train_data_ = {i: train_data[i] for i in range(self.args.num_clients)}
            with json_path.open("w") as f:
                json.dump(train_data_, f, cls=NpEncoder)
        else:
            raise TypeError("train_data must be dict or list")

        if test_data is not None:
            json_path = self.args.save_dir / "test_data.json"
            if type(test_data) is dict:
                with json_path.open("w") as f:
                    json.dump(test_data, f, cls=NpEncoder)
            elif type(test_data) is list:
                test_data_ = {i: test_data[i] for i in range(self.args.num_clients)}
                with json_path.open("w") as f:
                    json.dump(test_data_, f, cls=NpEncoder)
            else:
                raise TypeError("test_data must be dict or list")

    def add_default_args(self, args=None) -> Any:
        if args is None:
            if not hasattr(self, "default_args"):
                args = Config()
            else:
                args = deepcopy(self.default_args)
        # print("func args:", args)
        self.args.merge(args, overwrite=False)
        # print("func args:", self.args)
        # return self.args
