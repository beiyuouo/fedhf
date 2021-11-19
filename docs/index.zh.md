# FedHF

[FedHF](https://github.com/beiyuouo/FedHF) 是一个松耦合的，支持异构资源的，极其灵活的联邦学习框架。希望它能够帮助您更好更快的实现联邦学习相关算法。

*Accelerate your research*

![](https://img.shields.io/github/stars/beiyuouo/FedHF?style=flat-square) ![](https://img.shields.io/github/forks/beiyuouo/FedHF?style=flat-square) ![https://www.bj-yan.top/FedHF/](https://img.shields.io/badge/document-building-blue?style=flat-square) ![](https://img.shields.io/github/languages/code-size/beiyuouo/FedHF?style=flat-square) ![](https://img.shields.io/bitbucket/issues/beiyuouo/FedHF?style=flat-square) ![](https://img.shields.io/bitbucket/pr/beiyuouo/FedHF?style=flat-square) ![](https://img.shields.io/pypi/pyversions/fedhf?style=flat-square) ![](https://img.shields.io/pypi/l/fedhf?style=flat-square) 

## Features

- [x] 松耦合
- [x] 支持异构资源
- [x] 灵活的联邦学习框架
- [x] 支持异步的联邦学习
- [x] 实现了多种联邦学习算法

## Algorithms Supported

### Synchronous Aggregation

- [ ] **[FedAvg]** Communication-Efficient Learning of Deep Networks from Decentralized Data(*AISTAT*) [[paper]](https://arxiv.org/abs/1602.05629.pdf)

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

你可以在 [文档](https://www.bj-yan.top/FedHF/) 查看更多详细信息。

## Contributing

你可以在 [贡献指南](http://127.0.0.1:8000/FedHF/contributing/) 中查看更多详细信息。

## Citation

> *In progress*

## Licence

这个项目是 [Apache License Version 2.0](https://www.apache.org/licenses/LICENSE-2.0) 许可下的.

## Acknowledgement

非常感谢 [FedLab](https://github.com/SMILELab-FL/FedLab) 和 [FedML](https://github.com/FedML-AI/FedML) 作者们的优秀工作。
