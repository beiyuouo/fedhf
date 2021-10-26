#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   fedhf\api\opt\opt.py
@Time    :   2021/10/19 21:32:39
@Author  :   Bingjie Yan
@Email   :   bj.yan.pa@qq.com
"""

import argparse
import os


class opts(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        # high priority setting
        self.parser.add_argument('--from_file', default=None,
                                 help='load config from file.')

        # basic experiment setting
        self.parser.add_argument('--dataset', default='mnist',
                                 help='see fedhf/dataset for available datasets')
        self.parser.add_argument('--load_model', default='',
                                 help='path to pretrained model')
        self.parser.add_argument('--resume', action='store_true',
                                 help='resume an experiment. '
                                      'Reloaded the optimizer parameter and '
                                      'set load_model to model_last.pth '
                                      'in the exp dir if load_model is empty.')

        # system setting
        self.parser.add_argument('--gpus', default='0',
                                 help='-1 for CPU, use comma for multiple gpus')
        self.parser.add_argument('--num_workers', type=int, default=4,
                                 help='dataloader threads. 0 for single-thread.')
        self.parser.add_argument('--seed', type=int, default=233,
                                 help='random seed')

        # log setting
        self.parser.add_argument('--use_wandb', action='store_true',
                                 help='using wandb to store result')
        self.parser.add_argument('--project_name', default='fedhf', type=str,
                                 help='using for save result.')
        self.parser.add_argument('--log_name', default='logger', type=str,
                                 help='logger name')
        self.parser.add_argument('--log_file', default=None, type=str,
                                 help='where to save log')
        self.parser.add_argument('--log_level', default='debug', type=str,
                                 help='log level, it could be in [ error | warning | info | debug ]')

        # model setting
        self.parser.add_argument('--model', default='resnet',
                                 help='model for training')

        # training setting
        self.parser.add_argument('--optim', default='adam')
        self.parser.add_argument('--lr', type=float, default=1.25e-4,
                                 help='learning rate for batch size 32.')
        self.parser.add_argument('--check_point', type=str, default='50',
                                 help='when to save the model and result to disk.')
        self.parser.add_argument('--num_clients', type=int, default=3,
                                 help='clients number.')
        self.parser.add_argument('--num_local_epochs', type=int, default=3,
                                 help='local training epochs.')
        self.parser.add_argument('--batch_size', type=int, default=32,
                                 help='batch size')
        self.parser.add_argument('--num_rounds', type=int, default=10,
                                 help='server round.')

        # test setting

        # dataset setting

        # loss setting
        self.parser.add_argument('--loss_func', default='l1',
                                 help='regression loss: sl1 | l1 | l2')

        # custom dataset
        self.parser.add_argument('--custom_dataset_img_path', default='')
        self.parser.add_argument('--custom_dataset_ann_path', default='')

    def parse(self, args=''):
        if args == '':
            opt = self.parser.parse_args()
        else:
            opt = self.parser.parse_args(args)

        if opt.from_file:
            opt = self.load_from_file(args)

        opt.gpus_str = opt.gpus
        opt.gpus = [int(gpu) for gpu in opt.gpus.split(',')]
        opt.gpus = [i for i in range(
            len(opt.gpus))] if opt.gpus[0] >= 0 else [-1]

        opt.num_workers = max(opt.num_workers, 2 * len(opt.gpus))

        # make dirs
        # TODO

        if opt.resume and opt.load_model == '':
            opt.load_model = os.path.join(opt.save_dir, 'model_last.pth')
        return opt

    def load_from_file(self, args):
        # TODO
        pass
