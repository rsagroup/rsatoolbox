#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 10:17:51 2020

@author: heiko
"""

import unittest
import numpy as np


class TestRdmUtils(unittest.TestCase):

    def test_batch_to_vectors(self):
        from rsatoolbox.util.rdm_utils import batch_to_vectors
        dis = np.zeros((8, 5, 5))
        y, n_rdm, n_cond = batch_to_vectors(dis)
        assert y.shape[0] == 8
        assert y.shape[1] == 10
        assert n_rdm == 8
        assert n_cond == 5

    def test_batch_to_matrices(self):
        from rsatoolbox.util.rdm_utils import batch_to_matrices
        dis = np.zeros((8, 5, 5))
        y, n_rdm, n_cond = batch_to_matrices(dis)
        assert y.shape[0] == 8
        assert y.shape[1] == 5
        assert y.shape[2] == 5
        assert n_rdm == 8
        assert n_cond == 5

class TestPoolRDM(unittest.TestCase):

    def test_pool_standard(self):
        from rsatoolbox.rdm import RDMs
        from rsatoolbox.util.pooling import pool_rdm
        dissimilarities = np.random.rand(5, 10)
        rdms = RDMs(dissimilarities)
        for method in ['euclid', 'cosine', 'corr', 'cosine_cov', 'corr_cov',
                       'spearman', 'rho-a', 'tau-b', 'tau-a']:
            pooled_rdm = pool_rdm(rdms, method=method)
            self.assertEqual(pooled_rdm.n_cond, rdms.n_cond)
            self.assertEqual(pooled_rdm.n_rdm, 1)

    def test_pool_nan(self):
        from rsatoolbox.rdm import RDMs
        from rsatoolbox.util.pooling import pool_rdm
        dissimilarities = np.random.rand(5, 10)
        dissimilarities[:, 3] = np.nan
        rdms = RDMs(dissimilarities)
        for method in ['euclid', 'cosine', 'corr', 'cosine_cov', 'corr_cov',
                       'spearman', 'rho-a', 'tau-b', 'tau-a']:
            pooled_rdm = pool_rdm(rdms, method=method)
            self.assertEqual(pooled_rdm.n_cond, rdms.n_cond)
            self.assertEqual(pooled_rdm.n_rdm, 1)
            self.assertTrue(np.isnan(pooled_rdm.dissimilarities[0, 3]),
                            'nan got removed while pooling for %s' % method)
            self.assertFalse(np.isnan(pooled_rdm.dissimilarities[0, 4]),
                             'too many nans while pooling for %s' % method)

    def test_category_condition_idxs_2_categories_by_name(self):
        from rsatoolbox.util.rdm_utils import category_condition_idxs
        from rsatoolbox.rdm.rdms import RDMs

        n_rdm, n_cond = 4, 8
        dis = np.zeros((n_rdm, n_cond, n_cond))
        rdms = RDMs(dissimilarities=dis,
                    # 2 categories, P and Q
                    pattern_descriptors={'type': list('PPPPQQQQ')},
                    dissimilarity_measure='Euclidean',
                    descriptors={'subj': range(n_rdm)})
        categories = category_condition_idxs(rdms, 'type')

        self.assertEqual(categories, {
            'P': [0, 1, 2, 3],
            'Q': [4, 5, 6, 7],
        })

    def test_category_condition_idxs_2_categories_by_ints(self):
        from rsatoolbox.util.rdm_utils import category_condition_idxs
        from rsatoolbox.rdm.rdms import RDMs

        n_rdm, n_cond = 4, 8
        dis = np.zeros((n_rdm, n_cond, n_cond))
        rdms = RDMs(dissimilarities=dis,
                    dissimilarity_measure='Euclidean',
                    descriptors={'subj': range(n_rdm)})
        categories = category_condition_idxs(rdms, [1, 2, 1, 2, 1, 2, 1, 2])

        self.assertEqual(categories, {
            'Category 1': [0, 2, 4, 6],
            'Category 2': [1, 3, 5, 7],
        })
