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


class TestCalcRDM(unittest.TestCase):

    def setUp(self):
        measurements = np.random.rand(20, 5)
        measurements_deterministic = np.array([
            [0.11, 0.12, 0.21, 0.22, 0.30, 0.31],
            [0.13, 0.14, 0.24, 0.21, 0.29, 0.28],
            [0.10, 0.11, 0.24, 0.25, 0.32, 0.33],
        ]).T
        des = {'session': 0, 'subj': 0}
        obs_des = {'conds': np.array([0, 0, 1, 1, 2, 2, 2, 3, 4, 5,
                                      0, 0, 1, 1, 2, 2, 2, 3, 4, 5]),
                   'fold': np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                     1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
                   }
        obs_balanced = {'conds': np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4,
                                           0, 0, 1, 1, 2, 2, 3, 3, 4, 4]),
                        'fold': np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                          1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
                        }
        chn_des = {'rois': np.array(['V1', 'V1', 'IT', 'IT', 'V4'])}
        self.test_data = rsa.data.Dataset(
            measurements=measurements,
            descriptors=des,
            obs_descriptors=obs_des,
            channel_descriptors=chn_des
            )
        self.test_data_balanced = rsa.data.Dataset(
            measurements=measurements,
            descriptors=des,
            obs_descriptors=obs_balanced,
            channel_descriptors=chn_des
            )
        self.test_data_deterministic = rsa.data.Dataset(
            measurements=measurements_deterministic,
            descriptors=des,
            obs_descriptors=dict(
                conds=np.array([0, 0, 1, 1, 2, 2]),
            ),
            channel_descriptors=dict(feats=['v1', 'v2', 'v3'])
        )

    def test_calc_euclid_nconds(self):
        d = self.test_data
        rdm = rsr.calc_rdm_unbalanced(
            [d, d], descriptor='conds',
            method='euclidean')
        assert rdm.n_cond == 6
        assert rdm.n_rdm == 2

    @patch('rsatoolbox.rdm.calc._parse_input')
    def test_calc_euclid_as_scipy(self, _parse_input):
        from rsatoolbox.rdm import calc_rdm_unbalanced
        data = rsa.data.Dataset(
            np.random.rand(6, 5),
            descriptors={'session': 0, 'subj': 0},
            obs_descriptors={'conds': [0, 1, 2, 3, 4, 5]}
            )
        rdm_expected = pdist(data.measurements) ** 2 / 5
        rdms = calc_rdm_unbalanced(
            data,
            descriptor='conds',
            method='euclidean'
        )
        self.assertIsNone(
            assert_array_almost_equal(
                rdm_expected,
                rdms.dissimilarities.flatten()
                )
            )

    @patch('rsatoolbox.rdm.calc._parse_input')
    def test_calc_correlation(self, _parse_input):
        from rsatoolbox.rdm import calc_rdm_unbalanced
        data = rsa.data.Dataset(
            np.random.rand(6, 5),
            descriptors={'session': 0, 'subj': 0},
            obs_descriptors={'conds': [0, 1, 2, 3, 4, 5]}
            )
        rdm_expected = 1 - np.corrcoef(data.measurements)
        rdme = rsr.RDMs(
            dissimilarities=np.array([rdm_expected]),
            dissimilarity_measure='correlation',
            descriptors=data.descriptors)
        rdm = calc_rdm_unbalanced(
            data,
            descriptor='conds',
            method='correlation'
        )
        self.assertIsNone(
            assert_array_almost_equal(
                rdme.dissimilarities.flatten(),
                rdm.dissimilarities.flatten()
            )
        )

    def test_calc_list_descriptors(self):
        rdm = rsr.calc_rdm_unbalanced(
            [self.test_data, self.test_data, self.test_data],
            descriptor='conds',
            method='euclidean')
        assert np.all(rdm.rdm_descriptors['subj'] == np.array([0, 0, 0]))

    def test_calc_mahalanobis(self):
        rdm = rsr.calc_rdm_unbalanced(
            self.test_data, descriptor='conds',
            method='mahalanobis')
        assert rdm.n_cond == 6
        rdm_bal = rsr.calc_rdm_unbalanced(
            self.test_data, descriptor='conds',
            method='mahalanobis')
        rdm_check = rsr.calc_rdm(
            self.test_data, descriptor='conds',
            method='mahalanobis')
        assert_array_almost_equal(
            rdm_bal.dissimilarities.flatten(),
            rdm_check.dissimilarities.flatten()
            )

    def test_calc_crossnobis(self):
        rdm = rsr.calc_rdm_unbalanced(self.test_data,
                                      descriptor='conds',
                                      cv_descriptor='fold',
                                      method='crossnobis')
        assert rdm.n_cond == 6
        rdm_bal = rsr.calc_rdm_unbalanced(
            self.test_data, descriptor='conds',
            cv_descriptor='fold',
            method='crossnobis')
        rdm_check = rsr.calc_rdm(
            self.test_data, descriptor='conds',
            cv_descriptor='fold',
            method='crossnobis')
        assert_array_almost_equal(
            rdm_bal.dissimilarities.flatten(),
            rdm_check.dissimilarities.flatten()
            )

    def test_calc_crossnobis_no_descriptor(self):
        rdm = rsr.calc_rdm_unbalanced(self.test_data,
                                      descriptor='conds',
                                      method='crossnobis')
        assert rdm.n_cond == 6

    def test_calc_crossnobis_noise(self):
        noise = np.random.randn(10, 5)
        noise = np.matmul(noise.T, noise)
        rdm = rsr.calc_rdm_unbalanced(self.test_data,
                                      descriptor='conds', cv_descriptor='fold',
                                      noise=noise,
                                      method='crossnobis')
        assert rdm.n_cond == 6
        rdm_bal = rsr.calc_rdm_unbalanced(
            self.test_data_balanced, descriptor='conds',
            cv_descriptor='fold',
            noise=noise,
            method='crossnobis')
        rdm_check = rsr.calc_rdm(
            self.test_data_balanced, descriptor='conds',
            cv_descriptor='fold',
            noise=noise,
            method='crossnobis')
        assert_array_almost_equal(
            rdm_bal.dissimilarities.flatten(),
            rdm_check.dissimilarities.flatten()
            )

    def test_calc_poisson(self):
        """ for the poisson-KL the dissimilarities differ! This is explained
        in more detail in the demo on this computation"""
        rdm = rsr.calc_rdm_unbalanced(
            self.test_data,
            descriptor='conds',
            method='poisson'
        )
        assert rdm.n_cond == 6

    def test_calc_poisson_extreme_pairs(self):
        """Check the dissimilarities computed with the 'poisson' method

        The closest pair should be that between the condition 1 and itself
        The furthest pair should be that between condition 1 and condition 3
        """
        rdm = rsr.calc_rdm_unbalanced(
            self.test_data_deterministic,
            descriptor='conds',
            method='poisson'
        )
        rdm_array = squareform(rdm.get_vectors()[0, :])
        closest_pair_index = np.argmin(rdm_array)
        furthest_pair_index = np.argmax(rdm_array)
        self.assertEqual(closest_pair_index, 0)
        self.assertEqual(furthest_pair_index, 2)

    def test_calc_poisson_cv(self):
        rdm = rsr.calc_rdm_unbalanced(
            self.test_data,
            descriptor='conds',
            cv_descriptor='fold',
            method='poisson_cv')
        assert rdm.n_cond == 6
