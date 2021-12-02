# User Guide Overview

## Architecture

### Overview
[`FedHF`](https://github.com/beiyuouo/FedHF)是一个灵活的支持异构环境的联邦学习框架，提供了多种API可以快速实现联邦学习算法。

`FedHF`有着一个松耦合的结构，各个组件间尽可能的减少了引用关系，从而使得能够通过实现修改少量代码来实现新的算法。

首先，`FedHF`支持3种部署模式，分别是`simulated`、`distributed`和`standalone`。目前版本只实现了`simulated`模式。对于每种部署模式提供了3种方案，分别是`sync`、`async`和`tier-based`。

### `fedhf.core`

`FedHF`有三个高层核心模块位于`fedhf.core`，分别是协调器(`corrdinator`)，服务器(`server`)，客户端(`client`)。其中协调器包含服务器和客户端，用于管理总体进程，其职能还包括了解决数据划分方案和分布。服务器保存服务端模型，具有采样、聚合和验证的功能。客户端则主要承担具体训练和验证的任务。

### `fedhf.component`

接下来将依次介绍`fedhf.component`的所有组件。

### `fedhf.api`



### `deploy model`

- `simulated`: 是模拟联邦学习过程，方便部署和验证算法可行性，在`simulated`模式下，每一个协调器只需要一个服务器和一个客户端即可，所有的数据存储在协调器内，服务器和客户端只进行方法的操作，这与其他模式是不同的。

### `scheme`

- `sync`: 同步方案，服务器等待并收集所有被采样到的客户端上传的模型，并进行聚合，随后将聚合的模型发送给所有客户端。
- `async`: 异步方案，服务器收到任何客户端上传的模型后就会进行聚合，随和将该模型返还给客户端。
- `tier-based`: 层级方案，是一种同步异步的折中方案，对客户端的通信频率进行分级，分布实现同步和异步的方案。


### `fedhf.model`

### `fedhf.dataset`


## Docs structure
