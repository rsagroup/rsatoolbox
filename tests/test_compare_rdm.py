#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Tests for comparing RDMs

@author: heiko
"""

import unittest
import numpy as np
from numpy.testing import assert_array_almost_equal
import rsatoolbox as rsa


class TestCompareRDM(unittest.TestCase):

    def setUp(self):
        dissimilarities1 = np.random.rand(1, 15)
        des1 = {'session': 0, 'subj': 0}
        self.test_rdm1 = rsa.rdm.RDMs(
            dissimilarities=dissimilarities1,
            dissimilarity_measure='test',
            descriptors=des1)
        dissimilarities2 = np.random.rand(3, 15)
        des2 = {'session': 0, 'subj': 0}
        self.test_rdm2 = rsa.rdm.RDMs(
            dissimilarities=dissimilarities2,
            dissimilarity_measure='test',
            descriptors=des2
            )
        dissimilarities3 = np.random.rand(7, 15)
        des2 = {'session': 0, 'subj': 0}
        self.test_rdm3 = rsa.rdm.RDMs(
            dissimilarities=dissimilarities3,
            dissimilarity_measure='test',
            descriptors=des2
            )

    def test_compare_cosine(self):
        from rsatoolbox.rdm.compare import compare_cosine
        result = compare_cosine(self.test_rdm1, self.test_rdm1)
        assert_array_almost_equal(result, 1)
        result = compare_cosine(self.test_rdm1, self.test_rdm2)
        assert np.all(result < 1)

    def test_compare_cosine_cov(self):
        from rsatoolbox.rdm.compare import compare_cosine_cov_weighted
        result = compare_cosine_cov_weighted(self.test_rdm1,
                                             self.test_rdm1,
                                             sigma_k=np.eye(6))
        assert_array_almost_equal(result, 1)
        result = compare_cosine_cov_weighted(self.test_rdm1,
                                             self.test_rdm2,
                                             sigma_k=np.eye(6))
        assert np.all(result < 1)

    def test_compare_cosine_loop(self):
        from rsatoolbox.rdm.compare import compare_cosine
        result = compare_cosine(self.test_rdm2, self.test_rdm3)
        assert result.shape[0] == 3
        assert result.shape[1] == 7
        result_loop = np.zeros_like(result)
        d1 = self.test_rdm2.get_vectors()
        d2 = self.test_rdm3.get_vectors()
        for i in range(result_loop.shape[0]):
            for j in range(result_loop.shape[1]):
                result_loop[i, j] = (np.sum(d1[i] * d2[j])
                                     / np.sqrt(np.sum(d1[i] * d1[i]))
                                     / np.sqrt(np.sum(d2[j] * d2[j])))
        assert_array_almost_equal(result, result_loop)

    def test_compare_correlation(self):
        from rsatoolbox.rdm.compare import compare_correlation
        result = compare_correlation(self.test_rdm1, self.test_rdm1)
        assert_array_almost_equal(result, 1)
        result = compare_correlation(self.test_rdm1, self.test_rdm2)
        assert np.all(result < 1)

    def test_compare_correlation_cov(self):
        from rsatoolbox.rdm.compare import compare_correlation_cov_weighted
        result = compare_correlation_cov_weighted(self.test_rdm1,
                                                  self.test_rdm1)
        assert_array_almost_equal(result, 1)
        result = compare_correlation_cov_weighted(self.test_rdm1,
                                                  self.test_rdm2)
        assert np.all(result < 1)

    def test_compare_correlation_cov_sk(self):
        from rsatoolbox.rdm.compare import compare_correlation_cov_weighted
        result = compare_correlation_cov_weighted(self.test_rdm1,
                                                  self.test_rdm1,
                                                  sigma_k=np.eye(6))
        assert_array_almost_equal(result, 1)
        result = compare_correlation_cov_weighted(self.test_rdm1,
                                                  self.test_rdm2,
                                                  sigma_k=np.eye(6))
        assert np.all(result < 1)

    def test_compare_neg_riemannian_distance(self):
        from rsatoolbox.rdm.compare import compare_neg_riemannian_distance
        dataset1 = []
        for i_subj in range(1):
            dataset1.append(rsa.data.Dataset(np.random.rand(6, 20),
                                             descriptors={'subj': i_subj}))

        dataset2 = []
        for i_subj in range(5):
            dataset2.append(rsa.data.Dataset(np.random.rand(6, 20),
                                             descriptors={'subj': i_subj}))

        dataset3 = []
        for i_subj in range(7):
            dataset3.append(rsa.data.Dataset(np.random.rand(6, 20),
                                             descriptors={'subj': i_subj}))

        rdms1 = rsa.rdm.calc_rdm(dataset1, method='euclidean')
        rdms2 = rsa.rdm.calc_rdm(dataset2, method='euclidean')
        rdms3 = rsa.rdm.calc_rdm(dataset3, method='euclidean')

        result = compare_neg_riemannian_distance(rdms1, rdms1)
        assert_array_almost_equal(result, 0)
        result = compare_neg_riemannian_distance(rdms2, rdms3)
        assert result.shape[0] == 5
        assert result.shape[1] == 7
        assert np.all(result < 0)

    def test_compare_corr_loop(self):
        from rsatoolbox.rdm.compare import compare_correlation
        result = compare_correlation(self.test_rdm2, self.test_rdm3)
        assert result.shape[0] == 3
        assert result.shape[1] == 7
        result_loop = np.zeros_like(result)
        d1 = self.test_rdm2.get_vectors()
        d2 = self.test_rdm3.get_vectors()
        d1 = d1 - np.mean(d1, 1, keepdims=True)
        d2 = d2 - np.mean(d2, 1, keepdims=True)
        for i in range(result_loop.shape[0]):
            for j in range(result_loop.shape[1]):
                result_loop[i, j] = (np.sum(d1[i] * d2[j])
                                     / np.sqrt(np.sum(d1[i] * d1[i]))
                                     / np.sqrt(np.sum(d2[j] * d2[j])))
        assert_array_almost_equal(result, result_loop)

    def test_compare_spearman(self):
        from rsatoolbox.rdm.compare import compare_spearman
        result = compare_spearman(self.test_rdm1, self.test_rdm1)
        assert_array_almost_equal(result, 1)
        result = compare_spearman(self.test_rdm1, self.test_rdm2)
        assert np.all(result < 1)

    def test_compare_rho_a(self):
        from rsatoolbox.rdm.compare import compare_rho_a
        result = compare_rho_a(self.test_rdm1, self.test_rdm1)
        assert_array_almost_equal(result, 1)
        result = compare_rho_a(self.test_rdm1, self.test_rdm2)
        assert np.all(result < 1)

    def test_spearman_equal_scipy(self):
        from rsatoolbox.rdm.compare import _parse_input_rdms
        from rsatoolbox.rdm.compare import _all_combinations
        import scipy.stats
        from rsatoolbox.rdm.compare import compare_spearman

        def _spearman_r(vector1, vector2):
            """computes the spearman rank correlation between two vectors

            Args:
                vector1 (numpy.ndarray):
                    first vector
                vector1 (numpy.ndarray):
                    second vector
            Returns:
                corr (float):
                    spearman r

            """
            corr = scipy.stats.spearmanr(vector1, vector2).correlation
            return corr
        vector1, vector2, _ = _parse_input_rdms(self.test_rdm1, self.test_rdm2)
        sim = _all_combinations(vector1, vector2, _spearman_r)
        result = sim
        result2 = compare_spearman(self.test_rdm1, self.test_rdm2)
        assert_array_almost_equal(result, result2)

    def test_compare_kendall_tau(self):
        from rsatoolbox.rdm.compare import compare_kendall_tau
        result = compare_kendall_tau(self.test_rdm1, self.test_rdm1)
        assert_array_almost_equal(result, 1)
        result = compare_kendall_tau(self.test_rdm1, self.test_rdm2)
        assert np.all(result < 1)

    def test_compare_kendall_tau_a(self):
        from rsatoolbox.rdm.compare import compare_kendall_tau_a
        result = compare_kendall_tau_a(self.test_rdm1, self.test_rdm1)
        assert_array_almost_equal(result, 1)
        result = compare_kendall_tau_a(self.test_rdm1, self.test_rdm2)
        assert np.all(result < 1)

    def test_compare(self):
        from rsatoolbox.rdm.compare import compare
        result = compare(self.test_rdm1, self.test_rdm1)
        assert_array_almost_equal(result, 1)
        result = compare(self.test_rdm1, self.test_rdm2, method='corr')
        result = compare(self.test_rdm1, self.test_rdm2, method='corr_cov')
        result = compare(self.test_rdm1, self.test_rdm2, method='spearman')
        result = compare(self.test_rdm1, self.test_rdm2, method='cosine')
        result = compare(self.test_rdm1, self.test_rdm2, method='cosine_cov')
        result = compare(self.test_rdm1, self.test_rdm2, method='kendall')


class TestCompareRDMNaN(unittest.TestCase):

    def setUp(self):
        dissimilarities1 = np.random.rand(1, 15)
        des1 = {'session': 0, 'subj': 0}
        test_rdm1 = rsa.rdm.RDMs(
            dissimilarities=dissimilarities1,
            dissimilarity_measure='test',
            descriptors=des1)
        self.test_rdm1 = test_rdm1.subsample_pattern(
            'index', [0, 1, 1, 3, 4, 5])
        dissimilarities2 = np.random.rand(3, 15)
        des2 = {'session': 0, 'subj': 0}
        test_rdm2 = rsa.rdm.RDMs(
            dissimilarities=dissimilarities2,
            dissimilarity_measure='test',
            descriptors=des2
            )
        self.test_rdm2 = test_rdm2.subsample_pattern('index',
                                                     [0, 1, 1, 3, 4, 5])
        dissimilarities3 = np.random.rand(7, 15)
        des2 = {'session': 0, 'subj': 0}
        test_rdm3 = rsa.rdm.RDMs(
            dissimilarities=dissimilarities3,
            dissimilarity_measure='test',
            descriptors=des2
            )
        self.test_rdm3 = test_rdm3.subsample_pattern('index',
                                                     [0, 1, 1, 3, 4, 5])

    def test_compare_cosine(self):
        from rsatoolbox.rdm.compare import compare_cosine
        result = compare_cosine(self.test_rdm1, self.test_rdm1)
        assert_array_almost_equal(result, 1)
        result = compare_cosine(self.test_rdm1, self.test_rdm2)
        assert np.all(result < 1)

    def test_compare_cosine_cov(self):
        from rsatoolbox.rdm.compare import compare_cosine_cov_weighted
        result = compare_cosine_cov_weighted(self.test_rdm1,
                                             self.test_rdm1,
                                             sigma_k=np.eye(6))
        assert_array_almost_equal(result, 1)
        result = compare_cosine_cov_weighted(self.test_rdm1,
                                             self.test_rdm2,
                                             sigma_k=np.eye(6))
        assert np.all(result < 1)

    def test_compare_cosine_cov_sk(self):
        from rsatoolbox.rdm.compare import compare_cosine_cov_weighted
        result = compare_cosine_cov_weighted(self.test_rdm1,
                                             self.test_rdm2,
                                             sigma_k=None)
        result_1D = compare_cosine_cov_weighted(self.test_rdm1,
                                                self.test_rdm2,
                                                sigma_k=np.ones(6))
        result_2D = compare_cosine_cov_weighted(self.test_rdm1,
                                                self.test_rdm2,
                                                sigma_k=np.eye(6))
        assert_array_almost_equal(result, result_1D)
        assert_array_almost_equal(result, result_2D)

    def test_cosine_cov_consistency(self):
        from rsatoolbox.rdm.compare import _cosine_cov_weighted
        from rsatoolbox.rdm.compare import _cosine_cov_weighted_slow
        from rsatoolbox.rdm.compare import _parse_input_rdms
        vector1, vector2, nan_idx = _parse_input_rdms(self.test_rdm1,
                                                      self.test_rdm2)
        res_slow = _cosine_cov_weighted_slow(vector1, vector2, nan_idx=nan_idx)
        res = _cosine_cov_weighted(vector1, vector2, nan_idx=nan_idx)
        assert_array_almost_equal(res, res_slow)

    def test_compare_correlation(self):
        from rsatoolbox.rdm.compare import compare_correlation
        result = compare_correlation(self.test_rdm1, self.test_rdm1)
        assert_array_almost_equal(result, 1)
        result = compare_correlation(self.test_rdm1, self.test_rdm2)
        assert np.all(result < 1)

    def test_compare_correlation_cov(self):
        from rsatoolbox.rdm.compare import compare_correlation_cov_weighted
        result = compare_correlation_cov_weighted(self.test_rdm1,
                                                  self.test_rdm1)
        assert_array_almost_equal(result, 1)
        result = compare_correlation_cov_weighted(self.test_rdm1,
                                                  self.test_rdm2)
        assert np.all(result < 1)

    def test_compare_correlation_cov_sk(self):
        from rsatoolbox.rdm.compare import compare_correlation_cov_weighted
        result = compare_correlation_cov_weighted(self.test_rdm1,
                                                  self.test_rdm1,
                                                  sigma_k=np.eye(6))
        assert_array_almost_equal(result, 1)
        result = compare_correlation_cov_weighted(self.test_rdm1,
                                                  self.test_rdm2,
                                                  sigma_k=np.eye(6))
        assert np.all(result < 1)

    def test_compare_spearman(self):
        from rsatoolbox.rdm.compare import compare_spearman
        result = compare_spearman(self.test_rdm1, self.test_rdm1)
        assert_array_almost_equal(result, 1)
        result = compare_spearman(self.test_rdm1, self.test_rdm2)
        assert np.all(result < 1)

    def test_compare_rho_a(self):
        from rsatoolbox.rdm.compare import compare_rho_a
        result = compare_rho_a(self.test_rdm1, self.test_rdm1)
        result = compare_rho_a(self.test_rdm1, self.test_rdm2)
        assert np.all(result < 1)

    def test_spearman_equal_scipy(self):
        from rsatoolbox.rdm.compare import _parse_input_rdms
        from rsatoolbox.rdm.compare import _all_combinations
        import scipy.stats
        from rsatoolbox.rdm.compare import compare_spearman

        def _spearman_r(vector1, vector2):
            """computes the spearman rank correlation between two vectors

            Args:
                vector1 (numpy.ndarray):
                    first vector
                vector1 (numpy.ndarray):
                    second vector
            Returns:
                corr (float):
                    spearman r

            """
            corr = scipy.stats.spearmanr(vector1, vector2).correlation
            return corr
        vector1, vector2, _ = _parse_input_rdms(self.test_rdm1, self.test_rdm2)
        sim = _all_combinations(vector1, vector2, _spearman_r)
        result = sim
        result2 = compare_spearman(self.test_rdm1, self.test_rdm2)
        assert_array_almost_equal(result, result2)

    def test_compare_kendall_tau(self):
        from rsatoolbox.rdm.compare import compare_kendall_tau
        result = compare_kendall_tau(self.test_rdm1, self.test_rdm1)
        assert_array_almost_equal(result, 1)
        result = compare_kendall_tau(self.test_rdm1, self.test_rdm2)
        assert np.all(result < 1)

    def test_compare_kendall_tau_a(self):
        from rsatoolbox.rdm.compare import compare_kendall_tau_a
        result = compare_kendall_tau_a(self.test_rdm1, self.test_rdm1)
        result = compare_kendall_tau_a(self.test_rdm1, self.test_rdm2)
        assert np.all(result < 1)

    def test_compare(self):
        from rsatoolbox.rdm.compare import compare
        result = compare(self.test_rdm1, self.test_rdm1)
        assert_array_almost_equal(result, 1)
        result = compare(self.test_rdm1, self.test_rdm2, method='corr')
        result = compare(self.test_rdm1, self.test_rdm2, method='corr_cov')
        result = compare(self.test_rdm1, self.test_rdm2, method='spearman')
        result = compare(self.test_rdm1, self.test_rdm2, method='cosine')
        result = compare(self.test_rdm1, self.test_rdm2, method='cosine_cov')
        result = compare(self.test_rdm1, self.test_rdm2, method='kendall')


class TestCompareCov(unittest.TestCase):

    def setUp(self):
        dissimilarities1 = np.random.rand(1, 15)
        des1 = {'session': 0, 'subj': 0}
        self.test_rdm1 = rsa.rdm.RDMs(
            dissimilarities=dissimilarities1,
            dissimilarity_measure='test',
            descriptors=des1)
        dissimilarities2 = np.random.rand(3, 15)
        des2 = {'session': 0, 'subj': 0}
        self.test_rdm2 = rsa.rdm.RDMs(
            dissimilarities=dissimilarities2,
            dissimilarity_measure='test',
            descriptors=des2
            )
        dissimilarities3 = np.random.rand(7, 15)
        des2 = {'session': 0, 'subj': 0}
        self.test_rdm3 = rsa.rdm.RDMs(
            dissimilarities=dissimilarities3,
            dissimilarity_measure='test',
            descriptors=des2
            )

    def test_corr_identity_equal(self):
        from rsatoolbox.rdm.compare import compare
        result = compare(self.test_rdm1, self.test_rdm2, method='corr_cov')
        result_1D = compare(
            self.test_rdm1, self.test_rdm2, method='corr_cov',
            sigma_k=np.ones(6))
        result_2D = compare(
            self.test_rdm1, self.test_rdm2, method='corr_cov',
            sigma_k=np.eye(6))
        assert_array_almost_equal(result, result_1D)
        assert_array_almost_equal(result, result_2D)

    def test_cos_identity_equal(self):
        from rsatoolbox.rdm.compare import compare
        result = compare(self.test_rdm1, self.test_rdm2, method='cosine_cov')
        result_1D = compare(
            self.test_rdm1, self.test_rdm2, method='cosine_cov',
            sigma_k=np.ones(6))
        result_2D = compare(
            self.test_rdm1, self.test_rdm2, method='cosine_cov',
            sigma_k=np.eye(6))
        assert_array_almost_equal(result, result_1D)
        assert_array_almost_equal(result, result_2D)
