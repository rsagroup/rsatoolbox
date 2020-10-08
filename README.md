# Representational Similarity Analysis 3.0

[![Documentation Status](https://readthedocs.org/projects/rsa3/badge/?version=latest)](https://rsa3.readthedocs.io/en/latest/?badge=latest)
[![PyPI version](https://badge.fury.io/py/rsa3.svg)](https://badge.fury.io/py/rsa3)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/626ca9ec9f75485a9f73783c02710b1f)](https://www.codacy.com/gh/rsagroup/pyrsa?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=rsagroup/pyrsa&amp;utm_campaign=Badge_Grade)
[![CodeFactor](https://www.codefactor.io/repository/github/rsagroup/pyrsa/badge)](https://www.codefactor.io/repository/github/rsagroup/pyrsa)
[![codecov](https://codecov.io/gh/rsagroup/pyrsa/branch/master/graph/badge.svg)](https://codecov.io/gh/rsagroup/pyrsa)


Conceived during the RSA retreat 2019 in Blue Mountains,
this version replaces the 2013 version of pyrsa previously at ilogue/pyrsa.

[Documentation](https://rsa3.readthedocs.io/)


#### Getting Started

The easiest way to install pyrsa is with pip:

```sh
pip install rsa3
```

here is a simple code sample:

```python
import numpy, pyrsa
data = pyrsa.data.Dataset(numpy.random.rand(10, 5))
rdms = pyrsa.rdm.calc_rdm(data)
pyrsa.vis.show_rdm(rdms)
```