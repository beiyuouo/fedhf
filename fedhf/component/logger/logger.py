#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   fedhf\component\logger\logger.py
@Time    :   2021/10/19 10:45:58
@Author  :   Bingjie Yan
@Email   :   bj.yan.pa@qq.com
"""
import logging


import logging
import sys

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logging.getLogger().setLevel(logging.INFO)


class Logger(object):
    """record cmd info to file and print it to cmd at the same time

    Args:
        log_name (str): log name for output.
        log_file (str): a file path of log file.
    """

    def __init__(self, args):
        if args.log_name is not None:
            self.logger = logging.getLogger(args.log_name)
            self.name = args.log_name
        else:
            logging.getLogger().setLevel(logging.INFO)
            self.logger = logging
            self.name = "root"

        if args.log_file is not None:
            handler = logging.FileHandler(args.log_file, mode='w')
            handler.setLevel(level=logging.INFO)
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def info(self, log_str):
        """Print information to logger"""
        self.logger.info(log_str)

    def warning(self, warning_str):
        """Print warning to logger"""
        self.logger.warning(warning_str)
