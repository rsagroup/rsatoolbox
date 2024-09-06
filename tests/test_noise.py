#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Crossvalidation tests
@author: heiko
"""

import numpy as np
import unittest


class TestNoise(unittest.TestCase):
    """ crossvalidation tests
    """

    def setUp(self):
        self.mat2d = np.random.rand(250, 100)
        self.mat3d = np.random.rand(250, 100, 3)

    def test_2d(self):
        from rsatoolbox.data.noise import cov_from_residuals
        true_cov = np.cov(self.mat2d.T)
        rsa_cov = cov_from_residuals(self.mat2d, method="full")
        np.testing.assert_allclose(true_cov, rsa_cov)
        rsa_cov_diag = cov_from_residuals(self.mat2d, method="diag")
        np.testing.assert_allclose(np.diag(true_cov), np.diag(rsa_cov_diag))

    def test_3d(self):
        from rsatoolbox.data.noise import cov_from_residuals
        mat3d_to_2d = self.mat3d - np.mean(self.mat3d, 2, keepdims=True)
        mat3d_to_2d = mat3d_to_2d.transpose(0, 2, 1).reshape(
            mat3d_to_2d.shape[0] * mat3d_to_2d.shape[2], mat3d_to_2d.shape[1])
        true_cov = np.cov(mat3d_to_2d.T, ddof=self.mat3d.shape[0])
        rsa_cov = cov_from_residuals(self.mat3d, method="full")
        np.testing.assert_allclose(true_cov, rsa_cov)
        rsa_cov_diag = cov_from_residuals(self.mat3d, method="diag")
        np.testing.assert_allclose(np.diag(true_cov), np.diag(rsa_cov_diag))
