#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    :   fedhf\api\toolkits\plot.py
# @Time    :   2022-08-02 17:46:30
# @Author  :   Bingjie Yan
# @Email   :   bj.yan.pa@qq.com
# @License :   Apache License 2.0


import csv
import os
import numpy as np
import matplotlib.pyplot as plt
import click
import json


@click.group()
def plot():
    pass


@plot.command()
@click.argument("filename", type=click.Path(exists=True))
@click.argument(
    "output_dir",
    default=os.path.join("runs", "exp", "plot"),
    type=click.Path(exists=False),
)
def plot_timeline(filename, output_dir):
    # read from csv/log file
    # time, round, rank, event_group, event_type, event_content
    with open(filename, "r") as f:
        reader = csv.reader(f)
        data = list(reader)
    # format data
    time_data = []
    for row in data:
        row[0] = float(row[0])  # time
        row[1] = int(row[1])  # round
        row[2] = int(row[2])  # rank
        row[3] = str(row[3]).strip()  # event_group

        if "time" not in row[3]:
            continue

        row[4] = str(row[4]).strip()  # event_type
        row[5] = float(row[5])  # event_content

        row = np.array(row)
        time_data.append(row)

    # covert type
    time_data = np.array(time_data)
    time_data[:, 0] = time_data[:, 0].astype(np.float)
    time_data[:, 1] = time_data[:, 1].astype(np.int)
    time_data[:, 2] = time_data[:, 2].astype(np.int)
    time_data[:, 3] = time_data[:, 3].astype(np.str)
    time_data[:, 4] = time_data[:, 4].astype(np.str)

    x_limit = 100
    # normalize time
    time_data[:, 5] = (
        time_data[:, 5].astype(np.float) / time_data[:, 5].astype(np.float).max()
    )
    time_data[:, 5] = time_data[:, 5].astype(np.float) * x_limit

    # print(time_data)

    total_client = np.max(time_data[:, 2].astype(np.int)) + 1
    print("total client:", total_client)
    plt.figure(figsize=(12, 3))
    line_height = 0.4

    for client_idx in range(-1, total_client):
        # get data for this client
        plot_data = time_data[time_data[:, 2].astype(np.int) == client_idx]
        # plot
        last_time = 0
        y_min, y_max = (
            client_idx + 0.5 - line_height / 2,
            client_idx + 0.5 + line_height / 2,
        )
        for time_idx in range(plot_data.shape[0]):
            event_type = plot_data[time_idx, 4]
            if "end" in event_type:  # event type
                if "communication" in event_type:
                    color = "blue"
                elif "train" in event_type:
                    color = "green"
                elif "aggregation" in event_type:
                    color = "red"
                else:
                    color = "black"
                x_min, x_max = last_time, float(plot_data[time_idx, -1])
                x_min, x_max = x_min / x_limit, x_max / x_limit
                plt.axhspan(
                    ymin=y_min,
                    ymax=y_max,
                    xmin=x_min,
                    xmax=x_max,
                    color=color,
                    alpha=0.13,
                )
                print(
                    "client:",
                    client_idx,
                    "event:",
                    event_type,
                    "time:",
                    x_min,
                    x_max,
                    "color:",
                    color,
                    "y_min:",
                    y_min,
                    "y_max:",
                    y_max,
                )

            last_time = float(plot_data[time_idx, -1])

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.xlabel("timeline")
    plt.ylabel("client")
    plt.xlim(0, x_limit)
    y_keys = [f"client_{str(i)}" for i in range(total_client)]
    y_keys = ["server"] + y_keys
    plt.yticks(np.arange(0, total_client + 1) - 0.5, y_keys)
    plt.xticks(np.arange(0, x_limit, 10))
    plt.title("timeline")
    plt.savefig(os.path.join(output_dir, "timeline.png"))
    plt.show()


@plot.command()
@click.argument("filename", type=click.Path(exists=True))
def plot_data_partition(filename):
    with open(filename, "r") as f:
        data = json.load(f)
    plt.plot(data["time"], data["value"])
    plt.show()
