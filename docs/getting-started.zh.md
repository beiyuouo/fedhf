# 开始使用

## 安装

```sh
pip install fedhf

# If you want to use wandb to view log, please login first
wandb login
```

## 即刻开始使用

下面简单的程序提供了一个实现`FedAvg`的简单例子，完整的代码可以在[`fedhf.examples`](https://github.com/beiyuouo/FedHF/tree/main/fedhf/examples)中找到，关于详细的参数说明，可以在[`API-opts`](../user-guide/api#opts)中找到。


```python
args = opts().parse([
    '--deploy_mode', 'simulated', '--scheme', 'sync', '--model', 'cnn_mnist',
    '--dataset', 'mnist','--lr', '0.01', '--num_clients', '10','--num_rounds', '5',
    '--selector', 'random', '--select_ratio', '1', '--sampler', 'non-iid', 
    '--sampler_num_classes', '2', '--sampler_num_samples', '3000', '--agg',
    'fedavg'
])
coo = SimulatedSyncCoordinator(args)
coo.run()
```
