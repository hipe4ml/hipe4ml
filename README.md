![hipe4ml](./logo.svg)

![](https://github.com/hipe4ml/hipe4ml/workflows/Test%20package/badge.svg)
![](https://sonarcloud.io/api/project_badges/measure?project=hipe4ml_hipe4ml&metric=alert_status)
![](https://img.shields.io/github/license/hipe4ml/hipe4ml)
[![](https://img.shields.io/pypi/pyversions/hipe4ml.svg?longCache=True)](https://pypi.org/project/hipe4ml/)
[![](https://img.shields.io/pypi/v/hipe4ml.svg?maxAge=3600)](https://pypi.org/project/hipe4ml/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5070131.svg)](https://doi.org/10.5281/zenodo.5070131)

Minimal heavy ion physics environment for Machine Learning

To install:

```bash
pip install hipe4ml
```

**Mac OS X users:** the latest XGBoost version will require OpenMP installed. One easy way of getting it is using brew:

```bash
brew install libomp
```

# Documentation

<https://hipe4ml.github.io/>

# Cite

If you use this package for your analysis please consider citing it! You can use the following reference.

<https://doi.org/10.5281/zenodo.5070131>

# Contribute

## Install in-development package

To install the in-development package, from the repository base directory:

```bash
pip install -e .[dev]
```

## Run tests

If you have [Pylint](https://www.pylint.org/#install) and [Flake8](http://flake8.pycqa.org/en/latest/) on your development machine, you can run tests locally by using:

```bash
tests/run_tests.sh
```

in the repository base directory.

## Tutorials

If you want to get familiar with hipe4ml, the following tutorials are available:

| Type | Link |
| -------------- | ------------- |
| Binary classifier |  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hipe4ml/hipe4ml/blob/master/tutorials/hipe4ml_tutorial_binary.ipynb) |
