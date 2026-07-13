#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_util_matrix

@author: jdiedrichsen
"""

import unittest
import rsatoolbox.util as rsu
import numpy as np


class TestIndicator(unittest.TestCase):

    def test_indicator(self):
        a = np.array(range(0, 5))
        a = np.concatenate((a, a))
        X = rsu.matrix.indicator(a)
        n_row, n_col = X.shape
        self.assertEqual(n_row, 10)
        self.assertEqual(n_col, 5)
        self.assertEqual(X[0, 0], 1.0)

    def test_indicator_pos(self):
        a = np.array(range(0, 5))
        a = np.concatenate((a, a))
        X = rsu.matrix.indicator(a, positive=True)
        n_row, n_col = X.shape
        self.assertEqual(n_row, 10)
        self.assertEqual(n_col, 4)
        self.assertEqual(X[0, 0], 0.0)

    def test_pairwise(self):
        a = np.array(range(0, 5))
        X = rsu.matrix.pairwise_contrast(a)
        n_row, n_col = X.shape
        self.assertEqual(n_row, 10)
        self.assertEqual(n_col, 5)
        self.assertEqual(X[0, 0], 1.0)

    def test_centering(self):
        X = rsu.matrix.centering(10)
        n_row, n_col = X.shape
        self.assertEqual(n_row, 10)
        self.assertEqual(n_col, 10)

    def test_row_col_indicator_g(self):
        """ Tests the equivalence of the sparse and dense versions of row_col_indicator_g """
        for n_cond in [0, 1, 2, 3, 4, 5, 10, 20, 50, 100]:
            rowI_sparse, colI_sparse = rsu.matrix.row_col_indicator_g_sparse(n_cond)
            rowI_dense, colI_dense = rsu.matrix.row_col_indicator_g(n_cond)
            self.assertTrue(np.array_equal(rowI_sparse.toarray(), rowI_dense))
            self.assertTrue(np.array_equal(colI_sparse.toarray(), colI_dense))

if __name__ == '__main__':
    unittest.main()
