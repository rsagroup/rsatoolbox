#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 2020

@author: snormanhaignere
"""

import unittest
import numpy as np
from rsatoolbox.data.components import Components


class TestComponents(unittest.TestCase):
    def test_pca_reconstruction(self):
        X = np.random.randn(10, 100)
        components = Components()
        components.pca(X)
        assert np.allclose(components.reconstruct(), X)

    def test_fastica_recon_equals_pca_recon(self):
        X = np.random.randn(10, 100)
        X = X - np.mean(X, axis=1, keepdims=True)
        pca_components = Components()
        pca_components.pca(X, n_components=3)
        fastica_components = Components()
        fastica_components.fastica(X, n_components=3)
        assert np.allclose(pca_components.reconstruct(),
                           fastica_components.reconstruct())


if __name__ == '__main__':
    unittest.main()
