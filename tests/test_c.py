#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" tests for calculation of RDMs
"""

import unittest
from unittest.mock import patch
import numpy as np
from scipy.spatial.distance import pdist, squareform
import rsatoolbox as rsa
from rsatoolbox.cutil.similarity import similarity as similarity_c
from rsatoolbox.rdm.calc_unbalanced import calc_one_similarity as calc_one_similarity_c


class TestSimilarity(unittest.TestCase):

    def setUp(self):
        self.v_i = np.random.rand(21)
        self.v_j = np.random.rand(21)
        self.vec_i = np.array([0.11, 0.12, 0.22, 0.30, 0.31])
        self.vec_j = np.array([0.13, 0.14, 0.21, 0.29, 0.28])

    def test_basic(self):
        for i, method in enumerate(['euclidean', 'correlation', 'mahalanobis', 'poisson']):
            sim = similarity(self.vec_i, self.vec_j, method=method)
            sim_c, w = similarity_c(self.vec_i, self.vec_j, i + 1)
            self.assertAlmostEqual(sim, sim_c, None, 'C unequal to python for %s' % method)
        for i, method in enumerate(['euclidean', 'correlation', 'mahalanobis', 'poisson']):
            sim = similarity(self.v_i, self.v_j, method=method)
            sim_c, w = similarity_c(self.v_i, self.v_j, i + 1)
            self.assertAlmostEqual(sim, sim_c, None, 'C unequal to python for %s' % method)

    def test_noise(self):
        i = 2
        method = 'mahalanobis'
        noise = np.identity(5)
        noise[0, 0] = 2
        noise[0, 1] = 0.2
        noise[1, 0] = 0.2
        sim = similarity(self.vec_i, self.vec_j, method=method, noise=noise)
        sim_c, w = similarity_c(self.vec_i, self.vec_j, i + 1, noise=noise)
        self.assertAlmostEqual(sim, sim_c, None, 'C unequal to python for %s' % method)


class TestCalcOne(unittest.TestCase):

    def setUp(self):
        self.dat_i = np.random.rand(2, 21)
        self.dat_j = np.random.rand(3, 21)
        self.dat = np.concatenate((self.dat_i, self.dat_j), 0)
        self.data = rsa.data.Dataset(self.dat, obs_descriptors={'idx':[1,1,2,2,2]})

    def test_calc_one_similarity(self):
        d1 = self.data.subset_obs('idx', 1)
        d2 = self.data.subset_obs('idx', 2)
        for i, method in enumerate(['euclidean', 'correlation', 'mahalanobis', 'poisson']):
            sim, w = calc_one_similarity(self.data, method=method,
                                      descriptor='idx', i_des=1, j_des=2)
            sim_c, w_c = calc_one_similarity_c(d1,d2, np.array([0,1,2]), np.array([3,4]), method=method)
            self.assertAlmostEqual(sim, sim_c, None, 'C unequal to python for %s' % method)
            self.assertAlmostEqual(w, w_c, None, 'C unequal to python for %s weight' % method)


## Original Python version used as reference implementation:
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



def calc_one_similarity(dataset, descriptor, i_des, j_des,
                        method='euclidean',
                        noise=None, weighting='number',
                        prior_lambda=1, prior_weight=0.1,
                        cv_descriptor=None):
    """
    finds all pairs of vectors to be compared and calculates one distance

    Args:
        dataset (rsatoolbox.data.DatasetBase):
            dataset to extract from
        descriptor (String):
            key for the descriptor defining the conditions
        i_des : descriptor value
            the value of the first condition
        j_des : descriptor value
            the value of the second condition
        noise : numpy.ndarray (n_channels x n_channels), optional
            the covariance or precision matrix over channels
            necessary for calculation of mahalanobis distances

    Returns:
        (np.ndarray, np.ndarray) : (value, weight)
            value is the dissimilarity
            weight is the weight of the samples

    """
    data_i = dataset.subset_obs(descriptor, i_des)
    data_j = dataset.subset_obs(descriptor, j_des)
    values = []
    weights = []
    for i in range(data_i.n_obs):
        for j in range(data_j.n_obs):
            if cv_descriptor is None:
                accepted = True
            else:
                if (data_i.obs_descriptors[cv_descriptor][i]
                        == data_j.obs_descriptors[cv_descriptor][j]):
                    accepted = False
                else:
                    accepted = True
            if accepted:
                vec_i = data_i.measurements[i]
                vec_j = data_j.measurements[j]
                finite = np.isfinite(vec_i) & np.isfinite(vec_j)
                if np.any(finite):
                    if weighting == 'number':
                        weight = np.sum(finite)
                    elif weighting == 'equal':
                        weight = 1
                    sim = similarity(
                        vec_i[finite], vec_j[finite],
                        method,
                        noise=noise,
                        prior_lambda=prior_lambda,
                        prior_weight=prior_weight) \
                        / np.sum(finite)
                    values.append(sim)
                    weights.append(weight)
    weights = np.array(weights)
    values = np.array(values)
    if np.sum(weights) > 0:
        weight = np.sum(weights)
        value = np.sum(weights * values) / weight
    else:
        value = np.nan
        weight = 0
    return value, weight
