#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" tests for calculation of RDMs
"""

import unittest
from unittest.mock import patch
import numpy as np
from numpy.testing import assert_array_almost_equal
from scipy.spatial.distance import pdist, squareform
import rsatoolbox.rdm as rsr
import rsatoolbox as rsa
from rsatoolbox.rdm.calc_unbalanced import similarity
from rsatoolbox.cutil.similarity import similarity as similarity_c


class TestSimilarity(unittest.TestCase):

    def setUp(self):
        self.v_i = np.random.rand(21)
        self.v_j = np.random.rand(21)
        self.vec_i = np.array([0.11, 0.12, 0.21, 0.22, 0.30, 0.31])
        self.vec_j = np.array([0.13, 0.14, 0.24, 0.21, 0.29, 0.28])

    def test_basic(self):
        for i, method in enumerate(['euclidean', 'correlation', 'mahalanobis', 'poisson']):
            sim = similarity(self.vec_i, self.vec_j, method=method)
            sim_c = similarity_c(self.vec_i, self.vec_j, i + 1)
            self.assertAlmostEqual(sim, sim_c, None, 'C unequal to python for %s' % method)
        for i, method in enumerate(['euclidean', 'correlation', 'mahalanobis', 'poisson']):
            sim = similarity(self.v_i, self.v_j, method=method)
            sim_c = similarity_c(self.v_i, self.v_j, i + 1)
            self.assertAlmostEqual(sim, sim_c, None, 'C unequal to python for %s' % method)


def similarity(vec_i, vec_j, method, noise=None,
               prior_lambda=1, prior_weight=0.1):
    if method == 'euclidean':
        sim = np.sum(vec_i * vec_j)
    elif method == 'correlation':
        vec_i = vec_i - np.mean(vec_i)
        vec_j = vec_j - np.mean(vec_j)
        norm_i = np.sum(vec_i ** 2)
        norm_j = np.sum(vec_j ** 2)
        if (norm_i) > 0 and (norm_j > 0):
            sim = (np.sum(vec_i * vec_j)
                   / np.sqrt(norm_i) / np.sqrt(norm_j))
        else:
            sim = 1
        sim = sim * len(vec_i) / 2
    elif method in ['mahalanobis', 'crossnobis']:
        if noise is None:
            sim = similarity(vec_i, vec_j, 'euclidean')
        else:
            vec2 = (noise @ vec_j.T).T
            sim = np.sum(vec_i * vec2)
    elif method in ['poisson', 'poisson_cv']:
        vec_i = (vec_i + prior_lambda * prior_weight) \
            / (1 + prior_weight)
        vec_j = (vec_j + prior_lambda * prior_weight) \
            / (1 + prior_weight)
        sim = np.sum((vec_j - vec_i) * (np.log(vec_i) - np.log(vec_j))) / 2
    else:
        raise ValueError('dissimilarity method not recognized!')
    return sim
