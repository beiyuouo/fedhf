#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    :   fedhf\api\opt\opt.py
# @Time    :   2022-05-03 15:59:07
# @Author  :   Bingjie Yan
# @Email   :   bj.yan.pa@qq.com
# @License :   Apache License 2.0

import argparse
import os
import yaml

import torch
import torch.nn as nn
import numpy as np


class opts(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        # high priority setting
        self.parser.add_argument("--cfg", type=str, default="", help="config file path")

        # basic experiment setting
        self.parser.add_argument("--exp_name", default="exp", help="name of the experiment.")
        self.parser.add_argument("--exist_ok", action="store_true", help="allow to overwrite existing files.")
        self.parser.add_argument("--prj_name", default=None, help="name of the project.")
        self.parser.add_argument(
            "--deploy_mode", default="simulated", help="type of deployment. [ simulated, standalone, distributed ]"
        )
        self.parser.add_argument("--scheme", default="async", help="type of deployment. [ async, sync ]")

        # system setting
        self.parser.add_argument("--gpus", default="-1", help="-1 for cpu, use comma for multiple gpus")
        self.parser.add_argument("--num_workers", type=int, default=0, help="dataloader threads. 0 for single-thread.")

        # random setting
        self.parser.add_argument("--seed", type=int, default=233, help="random seed")

        # log setting
        self.parser.add_argument("--use_wandb", action="store_true", help="using wandb to store result")
        self.parser.add_argument("--wandb_log_client", action="store_true", help="log train on client or not")
        self.parser.add_argument("--wandb_reinit", action="store_true", help="reinit wandb")
        self.parser.add_argument("--log_name", default="logger", type=str, help="logger name")
        self.parser.add_argument("--log_dir", type=str, default=None, help="where to save the log.")
        self.parser.add_argument("--log_file", default=None, type=str, help="where to save log")
        self.parser.add_argument(
            "--log_level",
            default="debug",
            type=str,
            help="log level, it could be in [ error | warning | info | debug ]",
        )
        self.parser.add_argument("--eval_interval", type=int, default=5, help="evaluation interval")
        self.parser.add_argument(
            "--chkp_interval", type=int, default=50, help="when to save the model and result to disk."
        )
        self.parser.add_argument(
            "--save_dir", type=str, default="./chkp", help="where to save the model and result to disk."
        )

        # model setting
        self.parser.add_argument("--model", type=str, default="resnet", help="model name.")
        self.parser.add_argument("--model_pretrained", action="store_false", help="load pretrained model or not")
        self.parser.add_argument(
            "--model_dir", default="./model", type=str, help="path to download model"
        )  # Never used
        self.parser.add_argument(
            "--resume",
            default=None,
            type=str,
            help="resume an experiment. "
            "Reloaded the optimizer parameter and "
            "set load_model to args.name.pth "
            "in the save dir if load_model is empty.",
        )

        # data setting
        self.parser.add_argument("--dataset", default="mnist", help="see fedhf/dataset for available datasets")
        self.parser.add_argument("--resize", action="store_true", help="resize or not")
        self.parser.add_argument("--data_dir", default=os.path.join(".", "dataset"), help="dataset directory")
        self.parser.add_argument("--input_c", type=int, default=1, help="input channel")
        self.parser.add_argument("--image_size", type=int, default=224, help="image_size")
        self.parser.add_argument("--output_c", type=int, default=1, help="output channel")
        self.parser.add_argument("--num_classes", type=int, default=10, help="number of classes")

        self.parser.add_argument("--trainer", type=str, default="default_trainer", help="trainer.")
        self.parser.add_argument("--evaluator", type=str, default="default_evaluator", help="evaluator.")
        self.parser.add_argument("--optim", type=str, default="adam", help="optimizer.")
        self.parser.add_argument("--momentum", type=float, default=0.75, help="momentum.")
        self.parser.add_argument("--weight_decay", type=float, default=0.001, help="weight decay.")

        self.parser.add_argument("--lr", type=float, default=1e-2, help="learning rate.")
        self.parser.add_argument("--lr_scheduler", type=str, default=None, help="lr scheduler.")
        self.parser.add_argument("--lr_step", type=int, default=30, help="lr step.")

        self.parser.add_argument("--loss", type=str, default="ce", help="loss function.")
        self.parser.add_argument("--evaluate_on_client", action="store_true", help="evaluate on client")

        # federated setting
        self.parser.add_argument("--num_clients", type=int, default=100, help="clients number.")
        self.parser.add_argument("--num_clients_per_round", type=int, default=None, help="clients number per round.")
        self.parser.add_argument("--num_epochs", type=int, default=3, help="training epochs.")
        self.parser.add_argument("--batch_size", type=int, default=8, help="batch size")
        self.parser.add_argument("--num_rounds", type=int, default=5, help="server round.")

        self.parser.add_argument("--sampler", type=str, default="random", help="data sample strategy")

        self.parser.add_argument("--selector", type=str, default="random", help="client select strategy")
        self.parser.add_argument("--select_ratio", type=float, default=0.5, help="select ratio")
        self.parser.add_argument("--algor", type=str, default="fedavg", help="federated algorithm")

        # test setting
        self.parser.add_argument("--test", action="store_true", help="test mode")

    def parse(self, args=""):
        if args == "":
            opt = self.parser.parse_args()
        else:
            opt = self.parser.parse_args(args)

        return opt
