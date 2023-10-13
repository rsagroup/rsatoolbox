# Representational Similarity Analysis 3.0

[![Documentation Status](https://readthedocs.org/projects/rsatoolbox/badge/?version=latest)](https://rsatoolbox.readthedocs.io/en/latest/?badge=latest)
[![PyPI version](https://badge.fury.io/py/rsatoolbox.svg)](https://badge.fury.io/py/rsatoolbox)
[![Anaconda-Server Badge](https://anaconda.org/conda-forge/rsatoolbox/badges/version.svg)](https://anaconda.org/conda-forge/rsatoolbox)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/626ca9ec9f75485a9f73783c02710b1f)](https://www.codacy.com/gh/rsagroup/rsatoolbox?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=rsagroup/rsatoolbox&amp;utm_campaign=Badge_Grade)
[![CodeFactor](https://www.codefactor.io/repository/github/rsagroup/rsatoolbox/badge)](https://www.codefactor.io/repository/github/rsagroup/rsatoolbox)


Conceived during the RSA retreat 2019 in Blue Mountains.

[Documentation](https://rsatoolbox.readthedocs.io/)


#### Getting Started

To install the latest stable version of rsatoolbox with pip:

```sh
pip install rsatoolbox
```

or with conda:

```sh
conda install -c conda-forge rsatoolbox
```


here is a simple code sample:

```python
import numpy, rsatoolbox
data = rsatoolbox.data.Dataset(numpy.random.rand(10, 5))
rdms = rsatoolbox.rdm.calc_rdm(data)
rsatoolbox.vis.show_rdm(rdms)
```