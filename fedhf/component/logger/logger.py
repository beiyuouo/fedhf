#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   fedhf\component\logger\logger.py
@Time    :   2021-10-26 11:06:47
@Author  :   Bingjie Yan
@Email   :   bj.yan.pa@qq.com
@License :   Apache License 2.0
"""

import logging
import sys
import wandb

from .base_logger import BaseLogger, logger_map

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logging.getLogger().setLevel(logging.INFO)


class Logger(BaseLogger):

    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self, args):
        if args.log_level in logger_map:
            self.log_level = logger_map[args.log_level]
        else:
            raise "No such log level!"

        if args.log_name is not None:
            self.logger = logging.getLogger(args.log_name)
            self.name = args.log_name
        else:
            logging.getLogger().setLevel(logging.DEBUG)
            self.logger = logging
            self.name = "root"

        if args.log_file is not None:
            handler = logging.FileHandler(args.log_file, mode='w')
            handler.setLevel(level=logging.INFO)
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        if args.use_wandb:
            self.use_wandb = args.use_wandb
            wandb.init(project=args.project_name, config=args)

    def debug(self, log_str: str) -> None:
        self.logger.debug(log_str)

    def info(self, log_str: str) -> None:
        self.logger.info(log_str)

    def warning(self, log_str: str) -> None:
        self.logger.warning(log_str)

    def error(self, log_str: str) -> None:
        self.logger.error(log_str)

    def to_wandb(self, log_dict: dict) -> None:
        wandb.log(log_dict)
