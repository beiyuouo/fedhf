# Benchmark

All reproduceable benchmarks are listed below.

Training progress is shown in [wandb project](https://wandb.ai/bj-yan/fedhf-benchmark)

## MLP @ MNIST

| Method | Top-1 Accuracy | Round | Local Epoch | Clients | Distribution | Balance | Batch Size |
|:-----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|

## CNN @ MNIST

| Method | Top-1 Accuracy | Round | Local Epoch | Clients | Distribution | Balance | Batch Size |
|:-----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
| FedAsync | 93.17% | 400 | 3 | 100 | non-iid | balance, 2 classes | 50 |

## CNN @ CIFAR10

| Method | Top-1 Accuracy | Round | Local Epoch | Clients | Distribution | Balance | Batch Size |
|:-----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
| FedAsync | 42.25% | 2000 | 3 | 100 | non-iid | balance, 2 classes | 50 |
| FedAsync | 62.61% | 10000 | 5 | 100 | non-iid | balance, 2 classes | 50 |


## CNN @ FEMNIST

| Method | Top-1 Accuracy | Round | Local Epoch | Clients | Distribution | Balance | Batch Size |
|:-----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|

## RNN @ Shakespeare

| Method | Top-1 Accuracy | Round | Local Epoch | Clients | Distribution | Balance | Batch Size |
|:-----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
