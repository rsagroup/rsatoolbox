#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_data
Test for RDM class
@author: baihan
"""

import unittest
from unittest.mock import Mock, patch
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal
from scipy.spatial.distance import pdist, squareform
import pyrsa.rdm as rsr
import pyrsa as rsa



class TestRDM(unittest.TestCase):

    def test_rdm3d_init(self):
        dis = np.zeros((8, 5, 5))
        mes = "Euclidean"
        des = {'session': 0, 'subj': 0}
        rdms = rsr.RDMs(dissimilarities=dis,
                        dissimilarity_measure=mes,
                        descriptors=des)
        self.assertEqual(rdms.n_rdm, 8)
        self.assertEqual(rdms.n_cond, 5)

    def test_rdm2d_init(self):
        dis = np.zeros((8, 10))
        mes = "Euclidean"
        des = {'session': 0, 'subj': 0}
        rdms = rsr.RDMs(dissimilarities=dis,
                        dissimilarity_measure=mes,
                        descriptors=des)
        self.assertEqual(rdms.n_rdm, 8)
        self.assertEqual(rdms.n_cond, 5)

    def test_rdm3d_get_vectors(self):
        dis = np.zeros((8, 5, 5))
        mes = "Euclidean"
        des = {'session': 0, 'subj': 0}
        rdms = rsr.RDMs(dissimilarities=dis,
                        dissimilarity_measure=mes,
                        descriptors=des)
        v_rdms = rdms.get_vectors()
        self.assertEqual(v_rdms.shape[0], 8)
        self.assertEqual(v_rdms.shape[1], 10)

    def test_rdm2d_get_vectors(self):
        dis = np.zeros((8, 10))
        mes = "Euclidean"
        des = {'session': 0, 'subj': 0}
        rdms = rsr.RDMs(dissimilarities=dis,
                        dissimilarity_measure=mes,
                        descriptors=des)
        v_rdms = rdms.get_vectors()
        self.assertEqual(v_rdms.shape[0], 8)
        self.assertEqual(v_rdms.shape[1], 10)

    def test_rdm3d_get_matrices(self):
        dis = np.zeros((8, 5, 5))
        mes = "Euclidean"
        des = {'session': 0, 'subj': 0}
        rdms = rsr.RDMs(dissimilarities=dis,
                        dissimilarity_measure=mes,
                        descriptors=des)
        m_rdms = rdms.get_matrices()
        self.assertEqual(m_rdms.shape[0], 8)
        self.assertEqual(m_rdms.shape[1], 5)
        self.assertEqual(m_rdms.shape[2], 5)

    def test_rdm2d_get_matrices(self):
        dis = np.zeros((8, 10))
        mes = "Euclidean"
        des = {'session': 0, 'subj': 0}
        rdms = rsr.RDMs(dissimilarities=dis,
                        dissimilarity_measure=mes,
                        descriptors=des)
        m_rdms = rdms.get_matrices()
        self.assertEqual(m_rdms.shape[0], 8)
        self.assertEqual(m_rdms.shape[1], 5)
        self.assertEqual(m_rdms.shape[2], 5)

    def test_rdm_subset(self):
        dis = np.zeros((8, 10))
        mes = "Euclidean"
        des = {'subj': 0}
        rdm_des = {'session': np.array([0, 1, 2, 2, 4, 5, 6, 7])}
        rdms = rsr.RDMs(dissimilarities=dis,
                        rdm_descriptors=rdm_des,
                        dissimilarity_measure=mes,
                        descriptors=des)
        rdms_subset = rdms.subset('session', np.array([0, 1, 2]))
        self.assertEqual(rdms_subset.n_rdm, 4)
        self.assertEqual(rdms_subset.n_cond, 5)
        assert_array_equal(rdms_subset.rdm_descriptors['session'],
                           [0, 1, 2, 2])

    def test_rdm_subset_pattern(self):
        dis = np.zeros((8, 10))
        mes = "Euclidean"
        des = {'subj': 0}
        pattern_des = {'type': np.array([0, 1, 2, 2, 4])}
        rdms = rsr.RDMs(dissimilarities=dis,
                        pattern_descriptors=pattern_des,
                        dissimilarity_measure=mes,
                        descriptors=des)
        rdms_subset = rdms.subset_pattern('type', np.array([0, 1, 2]))
        self.assertEqual(rdms_subset.n_rdm, 8)
        self.assertEqual(rdms_subset.n_cond, 4)
        assert_array_equal(rdms_subset.pattern_descriptors['type'],
                           [0, 1, 2, 2])

    def test_rdm_subsample(self):
        dis = np.zeros((8, 10))
        mes = "Euclidean"
        des = {'subj': 0}
        rdm_des = {'session': np.array([0, 1, 2, 2, 4, 5, 6, 7])}
        rdms = rsr.RDMs(dissimilarities=dis,
                        rdm_descriptors=rdm_des,
                        dissimilarity_measure=mes,
                        descriptors=des)
        rdms_sample = rdms.subsample('session', np.array([0, 1, 2, 2]))
        self.assertEqual(rdms_sample.n_rdm, 6)
        self.assertEqual(rdms_sample.n_cond, 5)
        assert_array_equal(rdms_sample.rdm_descriptors['session'],
                           [0, 1, 2, 2, 2, 2])

    def test_rdm_subsample_pattern(self):
        dis = np.zeros((8, 10))
        mes = "Euclidean"
        des = {'subj': 0}
        pattern_des = {'type': np.array([0, 1, 2, 2, 4])}
        rdms = rsr.RDMs(dissimilarities=dis,
                        pattern_descriptors=pattern_des,
                        dissimilarity_measure=mes,
                        descriptors=des)
        rdms_sample = rdms.subsample_pattern('type',
                                             np.array([0, 1, 2, 2]))
        self.assertEqual(rdms_sample.n_rdm, 8)
        self.assertEqual(rdms_sample.n_cond, 6)
        assert_array_equal(rdms_sample.pattern_descriptors['type'],
                           [0, 1, 2, 2, 2, 2])

    def test_rdm_idx(self):
        dis = np.zeros((8, 10))
        mes = "Euclidean"
        des = {'subj': 0}
        pattern_des = {'type': np.array([0, 1, 2, 2, 4])}
        rdms = rsr.RDMs(dissimilarities=dis,
                        pattern_descriptors=pattern_des,
                        dissimilarity_measure=mes,
                        descriptors=des)
        rdms_sample = rdms[2]
        self.assertEqual(rdms_sample.n_rdm, 1)
        assert_array_equal(rdms_sample.dissimilarities[0], dis[2])
        rdms_sample = rdms[3, 4, 5]
        self.assertEqual(rdms_sample.n_rdm, 3)
        assert_array_equal(rdms_sample.dissimilarities[0], dis[3])

    def test_rank_transform(self):
        from pyrsa.rdm import rank_transform
        dis = np.zeros((8, 10))
        mes = "Euclidean"
        des = {'subj': 0}
        pattern_des = {'type': np.array([0, 1, 2, 2, 4])}
        rdm_des = {'session': np.array([0, 1, 2, 2, 4, 5, 6, 7])}
        rdms = rsr.RDMs(dissimilarities=dis,
                        rdm_descriptors=rdm_des,
                        pattern_descriptors=pattern_des,
                        dissimilarity_measure=mes,
                        descriptors=des)
        rank_rdm = rank_transform(rdms)
        assert rank_rdm.n_rdm == rdms.n_rdm
        assert rank_rdm.n_cond == rdms.n_cond

    def test_rdm_append(self):
        dis = np.zeros((8, 10))
        mes = "Euclidean"
        des = {'subj': 0}
        pattern_des = {'type': np.array([0, 1, 2, 2, 4])}
        rdm_des = {'session': np.array([0, 1, 2, 2, 4, 5, 6, 7])}
        rdms = rsr.RDMs(dissimilarities=dis,
                        pattern_descriptors=pattern_des,
                        dissimilarity_measure=mes,
                        descriptors=des,
                        rdm_descriptors=rdm_des)
        rdms.append(rdms)
        assert rdms.n_rdm == 16

    def test_concat(self):
        from pyrsa.rdm import concat
        dis = np.zeros((8, 10))
        dis2 = np.random.rand(8, 10)
        mes = "Euclidean"
        des = {'subj': 0}
        pattern_des = {'type': np.array([0, 1, 2, 2, 4])}
        rdm_des = {'session': np.array([0, 1, 2, 2, 4, 5, 6, 7])}
        rdms1 = rsr.RDMs(dissimilarities=dis,
                         pattern_descriptors=pattern_des,
                         dissimilarity_measure=mes,
                         descriptors=des,
                         rdm_descriptors=rdm_des)
        rdms2 = rsr.RDMs(dissimilarities=dis2,
                         pattern_descriptors=pattern_des,
                         dissimilarity_measure=mes,
                         descriptors=des,
                         rdm_descriptors=rdm_des)
        rdms = concat((rdms1, rdms2))
        assert rdms.n_rdm == 16

    def test_categorical_rdm(self):
        from pyrsa.rdm import get_categorical_rdm
        category_vector = [1, 2, 2, 3]
        rdm = get_categorical_rdm(category_vector)
        np.testing.assert_array_almost_equal(rdm.dissimilarities,
            np.array([[1., 1., 1., 0., 1., 1.]]))

    def test_reorder(self):
        from pyrsa.rdm import RDMs
        rdm = np.array([
            [0., 1., 2., 3.],
            [1., 0., 1., 2.],
            [2., 1., 0., 1.],
            [3., 2., 1., 0.]]
        )
        conds = ['a', 'b', 'c', 'd']
        rdms = RDMs(
            np.atleast_2d(squareform(rdm)),
            pattern_descriptors=dict(conds=conds)
        )
        conds_ordered = ['b', 'a', 'c', 'd']
        new_order = [conds.index(l) for l in conds_ordered]
        rdm_reordered = rdm[np.ix_(new_order, new_order)]
        rdms.reorder(new_order)
        assert_array_equal(
            np.atleast_2d(squareform(rdm_reordered)),
            rdms.dissimilarities
        )
        assert_array_equal(
            conds_ordered,
            rdms.pattern_descriptors.get('conds')
        )

    def test_sort_by(self):
        from pyrsa.rdm import RDMs
        rdm = np.array([
            [0., 1., 2., 3.],
            [1., 0., 1., 2.],
            [2., 1., 0., 1.],
            [3., 2., 1., 0.]]
        )
        conds = ['b', 'a', 'c', 'd']
        rdms = RDMs(
            np.atleast_2d(squareform(rdm)),
            pattern_descriptors=dict(conds=conds)
        )
        rdms.sort_by(conds='alpha')
        new_order = np.argsort(conds)
        rdm_reordered = rdm[np.ix_(new_order, new_order)]
        assert_array_equal(
            np.atleast_2d(squareform(rdm_reordered)),
            rdms.dissimilarities
        )
        assert_array_equal(
            sorted(conds),
            rdms.pattern_descriptors.get('conds')
        )


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
        rdm = rsr.calc_rdm(self.test_data, descriptor='conds',
                           method='euclidean')
        assert rdm.n_cond == 6

    def test_parse_input(self):
        from pyrsa.rdm.calc import _parse_input
        data = Mock()
        data.descriptors = {'session': 0, 'subj': 0}
        data.measurements = np.random.rand(6, 5)
        desc_true = [0, 1, 2, 3, 4, 5]
        measurements, desc, descriptor = _parse_input(data, None)
        assert descriptor == 'pattern'
        assert np.all(np.array(desc_true) == desc)
        assert np.all(data.measurements == measurements)

    @patch('pyrsa.rdm.calc._parse_input')
    def test_calc_euclid_as_scipy(self, _parse_input):
        from pyrsa.rdm import calc_rdm
        data = Mock()
        data.descriptors = {'session': 0, 'subj': 0}
        data.measurements = np.random.rand(6, 5)
        desc = [0, 1, 2, 3, 4, 5]
        _parse_input.return_value = (data.measurements, desc, 'conds')
        rdm_expected = pdist(data.measurements)**2/5
        rdms = calc_rdm(
            self.test_data,
            descriptor='conds',
            method='euclidean'
        )
        assert_array_almost_equal(
            rdm_expected,
            rdms.dissimilarities.flatten()
        )

    @patch('pyrsa.rdm.calc._parse_input')
    def test_calc_correlation(self, _parse_input):
        from pyrsa.rdm import calc_rdm
        data = Mock()
        data.descriptors = {'session': 0, 'subj': 0}
        data.measurements = np.random.rand(6, 5)
        desc = [0, 1, 2, 3, 4, 5]
        _parse_input.return_value = (data.measurements, desc, 'conds')
        rdm_expected = 1 - np.corrcoef(data.measurements)
        rdme = rsr.RDMs(
            dissimilarities=np.array([rdm_expected]),
            dissimilarity_measure='correlation',
            descriptors=data.descriptors)
        rdm = calc_rdm(
            self.test_data,
            descriptor='conds',
            method='correlation'
        )
        assert_array_almost_equal(
            rdme.dissimilarities.flatten(),
            rdm.dissimilarities.flatten()
        )

    def test_calc_mahalanobis(self):
        rdm = rsr.calc_rdm(self.test_data, descriptor='conds',
                           method='mahalanobis')
        assert rdm.n_cond == 6

    def test_calc_crossnobis(self):
        rdm = rsr.calc_rdm_crossnobis(self.test_data,
                                      descriptor='conds',
                                      cv_descriptor='fold')
        assert rdm.n_cond == 6

    def test_calc_crossnobis_no_descriptors(self):
        rdm = rsr.calc_rdm_crossnobis(self.test_data_balanced,
                                      descriptor='conds')
        assert rdm.n_cond == 5

    def test_calc_crossnobis_noise(self):
        noise = np.random.randn(10, 5)
        noise = np.matmul(noise.T, noise)
        rdm = rsr.calc_rdm_crossnobis(self.test_data_balanced,
                                      descriptor='conds',
                                      noise=noise)
        assert rdm.n_cond == 5

    def test_calc_crossnobis_noise_list(self):
        # generate two positive definite noise matricies
        noise = np.random.randn(2, 10, 5)
        noise = np.einsum('ijk,ijl->ikl', noise, noise)
        rdm = rsr.calc_rdm_crossnobis(self.test_data_balanced,
                                      cv_descriptor='fold',
                                      descriptor='conds',
                                      noise=noise)
        assert rdm.n_cond == 5
        # test with noise list
        noise = [noise[i] for i in range(len(noise))]
        rdm = rsr.calc_rdm_crossnobis(self.test_data_balanced,
                                      cv_descriptor='fold',
                                      descriptor='conds',
                                      noise=noise)
        assert rdm.n_cond == 5
        rdm = rsr.calc_rdm_crossnobis(self.test_data, cv_descriptor='fold',
                                      descriptor='conds', noise=noise)
        assert rdm.n_cond == 6

    def test_calc_poisson_6_conditions(self):
        rdm = rsr.calc_rdm(
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
        rdm = rsr.calc_rdm(
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
        rdm = rsr.calc_rdm(self.test_data,
                           descriptor='conds',
                           cv_descriptor='fold',
                           method='poisson_cv')
        assert rdm.n_cond == 6
        
class TestCalcRDMMovie(unittest.TestCase):
        
    def setUp(self):
        measurements_time = np.random.rand(20, 5, 15)
        tim_des = {'time': np.linspace(0,200, 15)}

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
        chn_des = {'rois': np.array(['sensor1', 'sensor2', 'sensor3', 'sensor5', 'sensor5'])}
        

        self.test_data_time = rsa.data.DatasetTime(measurements=measurements_time,
                               descriptors=des,
                               obs_descriptors=obs_des,
                               channel_descriptors=chn_des,
                               time_descriptors=tim_des
                               )
        self.test_data_time_balanced = rsa.data.DatasetTime(measurements=measurements_time,
                               descriptors=des,
                               obs_descriptors=obs_balanced,
                               channel_descriptors=chn_des,
                               time_descriptors=tim_des
                               )        
        
    def test_calc_rdm_movie_mahalanobis(self):
        rdm = rsr.calc_rdm_movie(self.test_data_time, descriptor='conds',
                           method='mahalanobis', time_descriptor = 'time')
        assert rdm.n_cond == 6
        assert len([r for r in rdm]) == 15
        assert rdm.rdm_descriptors['time'][0] == 0.0
        
    def test_calc_rdm_movie_euclidean(self):    
        rdm = rsr.calc_rdm_movie(self.test_data_time, descriptor='conds',
                           method='euclidean', time_descriptor = 'time')
        assert rdm.n_cond == 6
        assert len([r for r in rdm]) == 15
        assert rdm.rdm_descriptors['time'][0] == 0.0
    
    def test_calc_rdm_movie_correlation(self):            
        rdm = rsr.calc_rdm_movie(self.test_data_time, descriptor='conds',
                           method='correlation', time_descriptor = 'time')
        assert rdm.n_cond == 6
        assert len([r for r in rdm]) == 15
        assert rdm.rdm_descriptors['time'][0] == 0.0
        
    def test_calc_rdm_movie_crossnobis(self):
        rdm = rsr.calc_rdm_movie(self.test_data_time, descriptor='conds',
                           method='crossnobis', time_descriptor = 'time',
                                      cv_descriptor='fold')
        assert rdm.n_cond == 6
        assert len([r for r in rdm]) == 15
        assert rdm.rdm_descriptors['time'][0] == 0.0        
        
    def test_calc_rdm_movie_crossnobis_no_descriptors(self):
        rdm = rsr.calc_rdm_crossnobis(self.test_data_time_balanced,
                                      descriptor='conds')
        assert rdm.n_cond == 5

    def test_calc_rdm_movie_crossnobis_noise(self):
        noise = np.random.randn(10, 5)
        noise = np.matmul(noise.T, noise)
        rdm = rsr.calc_rdm_crossnobis(self.test_data_time_balanced,
                                      descriptor='conds',
                                      noise=noise)
        assert rdm.n_cond == 5
        
    def test_calc_rdm_movie_rdm_movie_poisson(self):        
        noise = np.random.randn(10, 5)
        noise = np.matmul(noise.T, noise)
        rdm = rsr.calc_rdm_movie(self.test_data_time_balanced,
                                      method='poisson',
                                      descriptor='conds',
                                      noise=noise)
        assert rdm.n_cond == 5        
        
        
    def test_calc_rdm_movie_binned(self):
        time = self.test_data_time.time_descriptors['time']
        bins = np.reshape(time, [5, 3])
        rdm = rsr.calc_rdm_movie(self.test_data_time, descriptor='conds',
                           method='mahalanobis', time_descriptor = 'time',
                           bins=bins)        
        assert rdm.n_cond == 6
        assert len([r for r in rdm]) == 5
        assert rdm.rdm_descriptors['time'][0] == np.mean(time[:3])
        

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
        from pyrsa.rdm.compare import compare_cosine
        result = compare_cosine(self.test_rdm1, self.test_rdm1)
        assert_array_almost_equal(result, 1)
        result = compare_cosine(self.test_rdm1, self.test_rdm2)
        assert np.all(result < 1)

    def test_compare_cosine_cov(self):
        from pyrsa.rdm.compare import compare_cosine_cov_weighted
        result = compare_cosine_cov_weighted(self.test_rdm1,
                                             self.test_rdm1,
                                             sigma_k=np.eye(6))
        assert_array_almost_equal(result, 1)
        result = compare_cosine_cov_weighted(self.test_rdm1,
                                             self.test_rdm2,
                                             sigma_k=np.eye(6))
        assert np.all(result < 1)

    def test_compare_cosine_loop(self):
        from pyrsa.rdm.compare import compare_cosine
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
        from pyrsa.rdm.compare import compare_correlation
        result = compare_correlation(self.test_rdm1, self.test_rdm1)
        assert_array_almost_equal(result, 1)
        result = compare_correlation(self.test_rdm1, self.test_rdm2)
        assert np.all(result < 1)

    def test_compare_correlation_cov(self):
        from pyrsa.rdm.compare import compare_correlation_cov_weighted
        result = compare_correlation_cov_weighted(self.test_rdm1,
                                                  self.test_rdm1)
        assert_array_almost_equal(result, 1)
        result = compare_correlation_cov_weighted(self.test_rdm1,
                                                  self.test_rdm2)
        assert np.all(result < 1)

    def test_compare_correlation_cov_sk(self):
        from pyrsa.rdm.compare import compare_correlation_cov_weighted
        result = compare_correlation_cov_weighted(self.test_rdm1,
                                                  self.test_rdm1,
                                                  sigma_k=np.eye(6))
        assert_array_almost_equal(result, 1)
        result = compare_correlation_cov_weighted(self.test_rdm1,
                                                  self.test_rdm2,
                                                  sigma_k=np.eye(6))
        assert np.all(result < 1)

    def test_compare_corr_loop(self):
        from pyrsa.rdm.compare import compare_correlation
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
                result_loop[i,j] = (np.sum(d1[i] * d2[j])
                                    / np.sqrt(np.sum(d1[i] * d1[i]))
                                    / np.sqrt(np.sum(d2[j] * d2[j])))
        assert_array_almost_equal(result, result_loop)

    def test_compare_spearman(self):
        from pyrsa.rdm.compare import compare_spearman
        result = compare_spearman(self.test_rdm1, self.test_rdm1)
        assert_array_almost_equal(result, 1)
        result = compare_spearman(self.test_rdm1, self.test_rdm2)
        assert np.all(result < 1)

    def test_compare_rho_a(self):
        from pyrsa.rdm.compare import compare_rho_a
        result = compare_rho_a(self.test_rdm1, self.test_rdm1)
        assert_array_almost_equal(result, 1)
        result = compare_rho_a(self.test_rdm1, self.test_rdm2)
        assert np.all(result < 1)

    def test_spearman_equal_scipy(self):
        from pyrsa.rdm.compare import _parse_input_rdms
        from pyrsa.rdm.compare import _all_combinations
        import scipy.stats
        from pyrsa.rdm.compare import compare_spearman

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
        vector1, vector2 = _parse_input_rdms(self.test_rdm1, self.test_rdm2)
        sim = _all_combinations(vector1, vector2, _spearman_r)
        result = sim
        result2 = compare_spearman(self.test_rdm1, self.test_rdm2)
        assert_array_almost_equal(result, result2)

    def test_compare_kendall_tau(self):
        from pyrsa.rdm.compare import compare_kendall_tau
        result = compare_kendall_tau(self.test_rdm1, self.test_rdm1)
        assert_array_almost_equal(result, 1)
        result = compare_kendall_tau(self.test_rdm1, self.test_rdm2)
        assert np.all(result < 1)

    def test_compare_kendall_tau_a(self):
        from pyrsa.rdm.compare import compare_kendall_tau_a
        result = compare_kendall_tau_a(self.test_rdm1, self.test_rdm1)
        assert_array_almost_equal(result, 1)
        result = compare_kendall_tau_a(self.test_rdm1, self.test_rdm2)
        assert np.all(result < 1)

    def test_compare(self):
        from pyrsa.rdm.compare import compare
        result = compare(self.test_rdm1, self.test_rdm1)
        assert_array_almost_equal(result, 1)
        result = compare(self.test_rdm1, self.test_rdm2, method='corr')
        result = compare(self.test_rdm1, self.test_rdm2, method='corr_cov')
        result = compare(self.test_rdm1, self.test_rdm2, method='spearman')
        result = compare(self.test_rdm1, self.test_rdm2, method='cosine')
        result = compare(self.test_rdm1, self.test_rdm2, method='cosine_cov')
        result = compare(self.test_rdm1, self.test_rdm2, method='kendall')


class TestSave(unittest.TestCase):
    def test_dict_conversion(self):
        dis = np.zeros((8, 10))
        mes = "Euclidean"
        des = {'subj': 0}
        pattern_des = {'type': np.array([0, 1, 2, 2, 4])}
        rdm_des = {'session': np.array([0, 1, 2, 2, 4, 5, 6, 7])}
        rdms = rsa.rdm.RDMs(
            dissimilarities=dis,
            pattern_descriptors=pattern_des,
            dissimilarity_measure=mes,
            descriptors=des,
            rdm_descriptors=rdm_des)
        rdm_dict = rdms.to_dict()
        rdms_loaded = rsa.rdm.rdms_from_dict(rdm_dict)
        assert rdms_loaded.n_cond == rdms.n_cond
        assert np.all(rdms_loaded.pattern_descriptors['type']
                      == pattern_des['type'])
        assert np.all(rdms_loaded.rdm_descriptors['session']
                      == rdm_des['session'])
        assert rdms_loaded.descriptors['subj'] == 0

    def test_save_load(self):
        import io
        f = io.BytesIO()  # Essentially a Mock file
        dis = np.zeros((8, 10))
        mes = "Euclidean"
        des = {'subj': 0}
        pattern_des = {'type': np.array([0, 1, 2, 2, 4])}
        rdm_des = {'session': np.array([0, 1, 2, 2, 4, 5, 6, 7])}
        rdms = rsa.rdm.RDMs(
            dissimilarities=dis,
            pattern_descriptors=pattern_des,
            dissimilarity_measure=mes,
            descriptors=des,
            rdm_descriptors=rdm_des)
        rdms.save(f, file_type='hdf5')
        rdms_loaded = rsa.rdm.load_rdm(f, file_type='hdf5')
        assert rdms_loaded.n_cond == rdms.n_cond
        assert np.all(rdms_loaded.pattern_descriptors['type']
                      == pattern_des['type'])
        assert np.all(rdms_loaded.rdm_descriptors['session']
                      == rdm_des['session'])
        assert rdms_loaded.descriptors['subj'] == 0


if __name__ == '__main__':
    unittest.main()
