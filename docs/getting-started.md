# Getting Started

## Installation

```sh
pip install fedhf

# If you want to use wandb to view log, please login first
wandb login
```

## Start to use

The following simple program provides a simple example of implementing `FedAvg`, the complete code can be found in `fedhf.examples`. For detailed parameter description, you can find it in [`API-opts`](../user-guide/api#opts). 

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
