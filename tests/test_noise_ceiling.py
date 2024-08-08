#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 16:59:33 2020

@author: heiko
"""

import unittest
import numpy as np
from parameterized import parameterized


class TestNoiseCeiling(unittest.TestCase):

    def setUp(self) -> None:
        from rsatoolbox.rdm import RDMs
        self.rng = np.random.default_rng(0)
        dis = self.rng.random((11, 10))  # 11 5x5 rdms
        mes = "Euclidean"
        des = {'subj': 0}
        rdm_des = {'session': np.array([1, 1, 2, 2, 4, 5, 6, 7, 7, 7, 7])}
        pattern_des = {'type': np.array([0, 1, 2, 2, 4])}
        self.rdms = RDMs(
            dissimilarities=dis,
            rdm_descriptors=rdm_des,
            pattern_descriptors=pattern_des,
            dissimilarity_measure=mes,
            descriptors=des
        )
        return super().setUp()

    def test_cv_noise_ceiling(self):
        from rsatoolbox.inference import cv_noise_ceiling
        from rsatoolbox.inference import sets_k_fold_rdm
        _, test_set, ceil_set = sets_k_fold_rdm(
            self.rdms, k_rdm=3, random=False)
        _, _ = cv_noise_ceiling(self.rdms, ceil_set, test_set, method='cosine')

    @parameterized.expand([
        ['cosine'],
        ['rho-a'],
        ['tau-a'],
        ['spearman'],
        ['corr'],
    ])
    def test_boot_noise_ceiling_runs_for_method(self, method):
        from rsatoolbox.inference import boot_noise_ceiling
        _, _ = boot_noise_ceiling(self.rdms, method=method)
