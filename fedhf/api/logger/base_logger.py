#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    :   fedhf\api\logger\base_logger.py
# @Time    :   2022-05-03 15:58:34
# @Author  :   Bingjie Yan
# @Email   :   bj.yan.pa@qq.com
# @License :   Apache License 2.0

from abc import ABC, abstractmethod

import logging

logger_map = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
}


class BaseLogger(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def error(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def warning(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def info(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def debug(self) -> None:
        raise NotImplementedError
