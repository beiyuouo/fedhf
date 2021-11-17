from setuptools import setup, find_packages
from os import path as os_path
import fedhf

this_directory = os_path.abspath(os_path.dirname(__file__))

setup(name="fedhf",
      version=fedhf.__version__,
      keywords=[
          "federated learning", "deep learning", "pytorch",
          "asynchronous federated learning", "wandb"
      ],
      author="Bingjie Yan",
      author_email="bj.yan.pa@qq.com",
      maintainer="Bingjie Yan",
      maintainer_email="bj.yan.pa@qq.com",
      description=
      "A Federated Learning Framework which is Heterogeneous and Flexible.",
      long_description=open(os_path.join(this_directory, "README.md")).read(),
      long_description_content_type="text/markdown",
      license="Apache-2.0 License",
      url="https://github.com/beiyuouo/FedHF",
      packages=find_packages(include=['fedhf', 'fedhf.*', 'LICENSE']),
      install_requires=[
          'torch>=1.8.2', 'torchvision>=0.9.2', 'numpy', 'tqdm', 'wandb'
      ],
      extras_require={
          "dev": ["nox", "pytest", "mkdocs"],
      },
      python_requires='>=3.6',
      classifiers=[
          'Topic :: Scientific/Engineering',
          'Topic :: Scientific/Engineering :: Artificial Intelligence',
          'Topic :: Software Development',
          'Topic :: Software Development :: Libraries',
          'Topic :: Software Development :: Libraries :: Python Modules',
          'License :: OSI Approved :: Apache Software License',
          'Programming Language :: Python :: 3'
      ])
