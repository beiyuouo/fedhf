#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   fedhf\examples\simulated-cnn-mnist\fedavg.py
@Time    :   2021-11-08 21:44:45
@Author  :   Bingjie Yan
@Email   :   bj.yan.pa@qq.com
@License :   Apache License 2.0
"""

from fedhf.api.opt.opt import opts
from fedhf.core.simulated import coordinator


def main():
    args = opts().parse()
    coo = coordinator(args)
    coo.run()


if __name__ == "__main__":
    main()
