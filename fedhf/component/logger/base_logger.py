#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   fedhf\component\logger\base_logger.py
@Time    :   2021-10-26 14:41:32
@Author  :   Bingjie Yan
@Email   :   bj.yan.pa@qq.com
@License :   Apache License 2.0
"""

from abc import ABC, abstractmethod

import logging


logger_map = {
    'debug': logging.DEBUG,
    'info': logging.INFO,
    'warning': logging.WARNING,
    'error': logging.ERROR
}


class BaseLogger(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def error(self, *args, **kwargs) -> None:
        raise NotImplementedError

    @abstractmethod
    def warning(self, *args, **kwargs) -> None:
        raise NotImplementedError

    @abstractmethod
    def info(self, *args, **kwargs) -> None:
        raise NotImplementedError

    @abstractmethod
    def debug(self, *args, **kwargs) -> None:
        raise NotImplementedError
