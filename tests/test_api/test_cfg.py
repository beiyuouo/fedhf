#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    :   tests\test_api\test_cfg.py
# @Time    :   2022-07-14 15:16:48
# @Author  :   Bingjie Yan
# @Email   :   bj.yan.pa@qq.com
# @License :   Apache License 2.0

import os
import fedhf
from fedhf import Config, opts
import yaml


class TestConfig(object):
    def test_config(self):
        cfg = Config([f"--cfg={os.path.join('tests', 'config', 'fedavg.yaml')}"], a=1, b=2, c=3)
        cfg = Config(cfg=f"{os.path.join('tests', 'config', 'fedavg.yaml')}", a=1, b=2, c=3)

        cfg_yaml = yaml.load(open(os.path.join("tests", "config", "fedavg.yaml"), "r"))

        assert cfg.a == 1
        assert cfg.b == 2
        assert cfg.c == 3
        assert cfg.cfg == os.path.join("tests", "config", "fedavg.yaml")
        # print(cfg)
        assert cfg.fedavg.ratio == cfg_yaml["fedavg"]["ratio"]
