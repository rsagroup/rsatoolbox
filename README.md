# Representational Similarity Analysis 3.0

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