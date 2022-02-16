# FedHF

<img src="https://raw.githubusercontent.com/beiyuouo/fedhf/main/docs/assets/logo.svg" alt="logo" style="width:100%;">

[FedHF](https://github.com/beiyuouo/fedhf) is a loosely coupled, **H**eterogeneous resource supported, and **F**lexible federated learning framework. 

*Accelerate your research*

![](https://img.shields.io/github/stars/beiyuouo/fedhf?style=flat-square) ![](https://img.shields.io/github/forks/beiyuouo/fedhf?style=flat-square) ![https://www.bj-yan.top/fedhf/](https://img.shields.io/badge/document-building-blue?style=flat-square) ![](https://img.shields.io/github/languages/code-size/beiyuouo/fedhf?style=flat-square) ![](https://img.shields.io/github/issues/beiyuouo/fedhf?style=flat-square) ![](https://img.shields.io/github/issues-pr/beiyuouo/fedhf?style=flat-square) ![](https://img.shields.io/pypi/pyversions/fedhf?style=flat-square) ![](https://img.shields.io/pypi/l/fedhf?style=flat-square) [![Downloads](https://pepy.tech/badge/fedhf)](https://pepy.tech/project/fedhf)

[![Discord](https://img.shields.io/discord/942746457855889469.svg?logo=discord&logoColor=white&logoWidth=20&labelColor=7289DA&label=Discord&style=flat-square)](https://discord.gg/AsvX9Q75)


## Features

- [x] Losely coupled
- [x] Heterogeneous resource supported
- [x] Flexible federated learning framework
- [x] Support for asynchronous aggregation
- [x] Support for multiple federated learning algorithms

## Algorithms Supported

### Synchronous Aggregation

- [x] **[FedAvg]** Communication-Efficient Learning of Deep Networks from Decentralized Data(*AISTAT*) [[paper]](https://arxiv.org/abs/1602.05629.pdf)

### Asynchronous Aggregation

- [x] **[FedAsync]** Asynchronous Federated Optimization(*OPT2020*) [[paper]](https://arxiv.org/abs/1903.03934)

### Tiered Aggregation

- [ ] **[TiFL]** TiFL: A Tier-based Federated Learning System (*HPDC 2020*) [[paper]](https://dl.acm.org/doi/abs/10.1145/3369583.3392686)

## Getting Start

```sh
pip install fedhf

# If you want to use wandb to view log, please login first
wandb login
```

You can see the [Document](https://www.bj-yan.top/fedhf/) for more details.

## Contributing

For more information, please see the [Contributing](https://www.bj-yan.top/fedhf/contributing/) page.

## Citation

> *In progress*

## Licence

This work is provided under [Apache License Version 2.0](https://www.apache.org/licenses/LICENSE-2.0).

## Acknowledgement

Many thanks to [FedLab](https://github.com/SMILELab-FL/FedLab) and [FedML](https://github.com/FedML-AI/FedML) for their great work.
