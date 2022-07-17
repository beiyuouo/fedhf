#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    :   fedhf\api\logger\logger.py
# @Time    :   2022-05-03 15:58:38
# @Author  :   Bingjie Yan
# @Email   :   bj.yan.pa@qq.com
# @License :   Apache License 2.0

import os
import logging
import sys
import time
from typing import Dict, Optional, Union

from .base_logger import BaseLogger, logger_map

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


class Logger(BaseLogger):
    class __Logger(BaseLogger):
        def __init__(self, args):
            if args.log_level in logger_map:
                self.log_level = logger_map[args.log_level]
            else:
                raise "No such log level!"

            if args.log_name is not None:
                self.log_name = args.log_name
            else:
                self.log_name = "root"

            self.logger = logging.getLogger(self.log_name)
            self.logger.setLevel(self.log_level)

            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )

            self.log_metric = args.log_metric

            file_handler = logging.FileHandler(args.log_file, mode="w")
            file_handler.setLevel(level=self.log_level)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

            # use streamHandler to print to stdout
            stream_handler = logging.StreamHandler(sys.stdout)
            stream_handler.setLevel(level=self.log_level)
            stream_handler.setFormatter(formatter)
            self.logger.addHandler(stream_handler)

            if args.use_wandb:
                import wandb

                self.use_wandb = args.use_wandb
                wandb.init(
                    project=args.project_name,
                    config=args,
                    reinit=args.wandb_reinit,
                    name=args.name,
                )

        def debug(self, log_str: str) -> None:
            self.logger.debug(log_str)

        def info(self, log_str: str) -> None:
            self.logger.info(log_str)

        def warning(self, log_str: str) -> None:
            self.logger.warning(log_str)

        def error(self, log_str: str) -> None:
            self.logger.error(log_str)

        def log(self, log_dict: dict, *args, **kwargs) -> None:
            # log one line in result.csv
            with open(self.log_file, "a") as f:
                f.write(str(log_dict) + "\n")

            if self.use_wandb:
                self.to_wandb(log_dict, *args, **kwargs)

        def to_wandb(self, log_dict: dict, *args, **kwargs) -> None:
            import wandb

            wandb.log(log_dict, *args, **kwargs)

        def log_metric(self, log_info: Union[Dict, str] = None) -> None:
            # log one line in result.csv
            with open(self.log_metric, "a") as f:
                f.write(str(log_info) + "\n")

    __instance = None

    def __new__(cls, args):
        if not cls.__instance:
            cls.__instance = Logger.__Logger(args)
        return cls.__instance

    def debug(self, log_str: str) -> None:
        self.__instance.debug(log_str)

    def info(self, log_str: str) -> None:
        self.__instance.info(log_str)

    def warning(self, log_str: str) -> None:
        self.__instance.warning(log_str)

    def error(self, log_str: str) -> None:
        self.__instance.error(log_str)

    def log(self, log_dict, *args, **kwargs) -> None:
        self.__instance.log(log_dict, args, kwargs)

    def log_metric(self, log_info, *args, **kwargs) -> None:
        self.__instance.log_metric(log_info, args, kwargs)

    def to_wandb(self, log_dict, *args, **kwargs) -> None:
        if self.__instance.use_wandb:
            self.__instance.to_wandb(log_dict, args, kwargs)
