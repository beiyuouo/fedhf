#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    :   fedhf\component\sampler\base_sampler.py
# @Time    :   2022-05-03 16:00:39
# @Author  :   Bingjie Yan
# @Email   :   bj.yan.pa@qq.com
# @License :   Apache License 2.0

from abc import ABC

from fedhf.api.utils.json_utils import NpEncoder


class BaseSampler(ABC):
    def __init__(self, args) -> None:
        self.args = args

    def sample(self):
        raise NotImplementedError

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
