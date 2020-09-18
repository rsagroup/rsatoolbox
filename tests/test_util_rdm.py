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
        from pyrsa.util.rdm_utils import batch_to_vectors
        dis = np.zeros((8,5,5))
        y, n_rdm, n_cond = batch_to_vectors(dis)
        assert y.shape[0] == 8
        assert y.shape[1] == 10
        assert n_rdm == 8
        assert n_cond == 5

    def test_batch_to_matrices(self):
        from pyrsa.util.rdm_utils import batch_to_matrices
        dis = np.zeros((8,5,5))
        y, n_rdm, n_cond = batch_to_matrices(dis)
        assert y.shape[0] == 8
        assert y.shape[1] == 5
        assert y.shape[2] == 5
        assert n_rdm == 8
        assert n_cond == 5

    def test_category_selector_to_names_and_idxs_2_categories_by_name(self):
        from pyrsa.util.rdm_utils import category_selector_to_names_and_idxs
        from pyrsa.rdm.rdms import RDMs

        n_rdm, n_cond = 4, 8
        dis = np.zeros((n_rdm, n_cond, n_cond))
        rdms = RDMs(dissimilarities=dis,
                    # 2 categories, P and Q
                    pattern_descriptors={'type': list('PPPPQQQQ')},
                    dissimilarity_measure='Euclidean',
                    descriptors={'subj': range(n_rdm)})
        category_names, condition_idxs = category_selector_to_names_and_idxs(rdms, 'type')

        self.assertEqual(category_names, ['P', 'Q'])  # Names correct
        self.assertEqual(condition_idxs, {  # Indices correct
            'P': [0, 1, 2, 3],
            'Q': [4, 5, 6, 7],
        })

    def test_category_selector_to_names_and_idxs_2_categories_by_ints(self):
        from pyrsa.util.rdm_utils import category_selector_to_names_and_idxs
        from pyrsa.rdm.rdms import RDMs

        n_rdm, n_cond = 4, 8
        dis = np.zeros((n_rdm, n_cond, n_cond))
        rdms = RDMs(dissimilarities=dis,
                    dissimilarity_measure='Euclidean',
                    descriptors={'subj': range(n_rdm)})
        category_names, condition_idxs = category_selector_to_names_and_idxs(rdms, [1, 2, 1, 2, 1, 2, 1, 2])

        self.assertEqual(category_names, ['Category 1', 'Category 2'])  # Names correct
        self.assertEqual(condition_idxs, {  # Indices correct
            'Category 1': [0, 2, 4, 6],
            'Category 2': [1, 3, 5, 7],
        })
