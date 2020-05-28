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
from numpy.testing import assert_array_almost_equal
from numpy.testing import assert_array_equal
from scipy.spatial.distance import pdist
import pyrsa.rdm as rsr
import pyrsa as rsa

class TestRDM(unittest.TestCase): 
    
    def test_rdm3d_init(self):
        dis = np.zeros((8,5,5))
        mes = "Euclidean"
        des = {'session':0, 'subj':0}
        rdms = rsr.RDMs(dissimilarities=dis,
                        dissimilarity_measure=mes,
                        descriptors=des)
        self.assertEqual(rdms.n_rdm,8)
        self.assertEqual(rdms.n_cond,5)

    def test_rdm2d_init(self):
        dis = np.zeros((8,10))
        mes = "Euclidean"
        des = {'session':0, 'subj':0}
        rdms = rsr.RDMs(dissimilarities=dis,
                        dissimilarity_measure=mes,
                        descriptors=des)
        self.assertEqual(rdms.n_rdm,8)
        self.assertEqual(rdms.n_cond,5)

    def test_rdm3d_get_vectors(self):
        dis = np.zeros((8,5,5))
        mes = "Euclidean"
        des = {'session':0, 'subj':0}
        rdms = rsr.RDMs(dissimilarities=dis,
                        dissimilarity_measure=mes,
                        descriptors=des)
        v_rdms = rdms.get_vectors()
        self.assertEqual(v_rdms.shape[0],8)
        self.assertEqual(v_rdms.shape[1],10)

    def test_rdm2d_get_vectors(self):
        dis = np.zeros((8,10))
        mes = "Euclidean"
        des = {'session':0, 'subj':0}
        rdms = rsr.RDMs(dissimilarities=dis,
                        dissimilarity_measure=mes,
                        descriptors=des)
        v_rdms = rdms.get_vectors()
        self.assertEqual(v_rdms.shape[0],8)
        self.assertEqual(v_rdms.shape[1],10)

    def test_rdm3d_get_matrices(self):
        dis = np.zeros((8,5,5))
        mes = "Euclidean"
        des = {'session':0, 'subj':0}
        rdms = rsr.RDMs(dissimilarities=dis,
                        dissimilarity_measure=mes,
                        descriptors=des)
        m_rdms = rdms.get_matrices()
        self.assertEqual(m_rdms.shape[0],8)
        self.assertEqual(m_rdms.shape[1],5)
        self.assertEqual(m_rdms.shape[2],5)

    def test_rdm2d_get_matrices(self):
        dis = np.zeros((8,10))
        mes = "Euclidean"
        des = {'session':0, 'subj':0}
        rdms = rsr.RDMs(dissimilarities=dis,
                        dissimilarity_measure=mes,
                        descriptors=des)
        m_rdms = rdms.get_matrices()
        self.assertEqual(m_rdms.shape[0],8)
        self.assertEqual(m_rdms.shape[1],5)
        self.assertEqual(m_rdms.shape[2],5)

    def test_rdm_subset(self):
        dis = np.zeros((8,10))
        mes = "Euclidean"
        des = {'subj':0}
        rdm_des = {'session':np.array([0,1,2,2,4,5,6,7])}
        rdms = rsr.RDMs(dissimilarities=dis,
                        rdm_descriptors=rdm_des,
                        dissimilarity_measure=mes,
                        descriptors=des)
        rdms_subset = rdms.subset('session',np.array([0,1,2]))
        self.assertEqual(rdms_subset.n_rdm,4)
        self.assertEqual(rdms_subset.n_cond,5)
        assert_array_equal(rdms_subset.rdm_descriptors['session'],[0,1,2,2])

    def test_rdm_subset_pattern(self):
        dis = np.zeros((8,10))
        mes = "Euclidean"
        des = {'subj':0}
        pattern_des = {'type':np.array([0,1,2,2,4])}
        rdms = rsr.RDMs(dissimilarities=dis,
                        pattern_descriptors=pattern_des,
                        dissimilarity_measure=mes,
                        descriptors=des)
        rdms_subset = rdms.subset_pattern('type',np.array([0,1,2]))
        self.assertEqual(rdms_subset.n_rdm,8)
        self.assertEqual(rdms_subset.n_cond,4)
        assert_array_equal(rdms_subset.pattern_descriptors['type'],[0,1,2,2])

    def test_rdm_subsample(self):
        dis = np.zeros((8,10))
        mes = "Euclidean"
        des = {'subj':0}
        rdm_des = {'session':np.array([0,1,2,2,4,5,6,7])}
        rdms = rsr.RDMs(dissimilarities=dis,
                        rdm_descriptors=rdm_des,
                        dissimilarity_measure=mes,
                        descriptors=des)
        rdms_sample = rdms.subsample('session', np.array([0,1,2,2]))
        self.assertEqual(rdms_sample.n_rdm,6)
        self.assertEqual(rdms_sample.n_cond,5)
        assert_array_equal(rdms_sample.rdm_descriptors['session'],
                           [0,1,2,2,2,2])

    def test_rdm_subsample_pattern(self):
        dis = np.zeros((8,10))
        mes = "Euclidean"
        des = {'subj':0}
        pattern_des = {'type':np.array([0,1,2,2,4])}
        rdms = rsr.RDMs(dissimilarities=dis,
                        pattern_descriptors=pattern_des,
                        dissimilarity_measure=mes,
                        descriptors=des)
        rdms_sample = rdms.subsample_pattern('type',np.array([0,1,2,2]))
        self.assertEqual(rdms_sample.n_rdm,8)
        self.assertEqual(rdms_sample.n_cond,6)
        assert_array_equal(rdms_sample.pattern_descriptors['type'],[0,1,2,2,2,2])

    def test_rdm_idx(self):
        dis = np.zeros((8,10))
        mes = "Euclidean"
        des = {'subj':0}
        pattern_des = {'type':np.array([0,1,2,2,4])}
        rdms = rsr.RDMs(dissimilarities=dis,
                        pattern_descriptors=pattern_des,
                        dissimilarity_measure=mes,
                        descriptors=des)
        rdms_sample = rdms[2]
        self.assertEqual(rdms_sample.n_rdm,1)
        assert_array_equal(rdms_sample.dissimilarities[0], dis[2])
        rdms_sample = rdms[3,4,5]
        self.assertEqual(rdms_sample.n_rdm,3)
        assert_array_equal(rdms_sample.dissimilarities[0], dis[3])

    def test_rank_transform(self):
        from pyrsa.rdm import rank_transform
        dis = np.zeros((8,10))
        mes = "Euclidean"
        des = {'subj':0}
        pattern_des = {'type':np.array([0,1,2,2,4])}
        rdm_des = {'session':np.array([0,1,2,2,4,5,6,7])}
        rdms = rsr.RDMs(dissimilarities=dis,
                        rdm_descriptors=rdm_des,
                        pattern_descriptors=pattern_des,
                        dissimilarity_measure=mes,
                        descriptors=des)
        rank_rdm = rank_transform(rdms)
        assert rank_rdm.n_rdm == rdms.n_rdm
        assert rank_rdm.n_cond == rdms.n_cond
        
        
        
    def test_rdm_append(self):
        dis = np.zeros((8,10))
        mes = "Euclidean"
        des = {'subj':0}
        pattern_des = {'type':np.array([0,1,2,2,4])}
        rdm_des = {'session':np.array([0,1,2,2,4,5,6,7])}
        rdms = rsr.RDMs(dissimilarities=dis,
                        pattern_descriptors=pattern_des,
                        dissimilarity_measure=mes,
                        descriptors=des,
                        rdm_descriptors=rdm_des)
        rdms.append(rdms)
        assert rdms.n_rdm == 16
    
    def test_concat(self):
        from pyrsa.rdm import concat
        dis = np.zeros((8,10))
        dis2 = np.random.rand(8,10)
        mes = "Euclidean"
        des = {'subj':0}
        pattern_des = {'type':np.array([0,1,2,2,4])}
        rdm_des = {'session':np.array([0,1,2,2,4,5,6,7])}
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
        rdms = concat((rdms1,rdms2))
        assert rdms.n_rdm == 16

    def test_categorical_rdm(self):
        from pyrsa.rdm import get_categorical_rdm
        category_vector = [1,2,2,3]
        rdm = get_categorical_rdm(category_vector)
        np.testing.assert_array_almost_equal(rdm.dissimilarities,
            np.array([[1., 1., 1., 0., 1., 1.]]))

class TestCalcRDM(unittest.TestCase): 
    
    def setUp(self):
        measurements = np.random.rand(20,5)
        des = {'session':0,'subj':0}
        obs_des = {'conds':np.array([0,0,1,1,2,2,2,3,4,5,0,0,1,1,2,2,2,3,4,5]),
                   'fold':np.array([0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1])
                   }
        chn_des = {'rois':np.array(['V1','V1','IT','IT','V4'])}
        self.test_data = rsa.data.Dataset(measurements=measurements,
                           descriptors=des,
                           obs_descriptors=obs_des,
                           channel_descriptors=chn_des
                           )

    def test_calc_euclid_nconds(self):
        rdm = rsr.calc_rdm(self.test_data, descriptor = 'conds', method = 'euclidean')
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
        rdme = rsr.RDMs(dissimilarities=np.array([rdm_expected]),
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
        rdm = rsr.calc_rdm(self.test_data, descriptor = 'conds', method = 'mahalanobis')
        assert rdm.n_cond == 6
        
    def test_calc_crossnobis(self):
        rdm = rsr.calc_rdm_crossnobis(self.test_data, descriptor = 'conds', cv_descriptor = 'fold')
        assert rdm.n_cond == 6


class TestCompareRDM(unittest.TestCase): 
    
    def setUp(self):
        dissimilarities1 = np.random.rand(1,15)
        des1 = {'session':0,'subj':0}
        self.test_rdm1 = rsa.rdm.RDMs(dissimilarities=dissimilarities1,
                           dissimilarity_measure='test',
                           descriptors=des1
                           )
        dissimilarities2 = np.random.rand(3,15)
        des2 = {'session':0,'subj':0}
        self.test_rdm2 = rsa.rdm.RDMs(dissimilarities=dissimilarities2,
                           dissimilarity_measure='test',
                           descriptors=des2
                           )
        dissimilarities3 = np.random.rand(7,15)
        des2 = {'session':0,'subj':0}
        self.test_rdm3 = rsa.rdm.RDMs(dissimilarities=dissimilarities3,
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
                result_loop[i,j] = (np.sum(d1[i] * d2[j]) 
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
        assert_array_almost_equal(result, 0)
        result = compare_rho_a(self.test_rdm1, self.test_rdm2)
        assert np.all(result>0)
        
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
        assert_array_almost_equal(result,result2)
        
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
        dis = np.zeros((8,10))
        mes = "Euclidean"
        des = {'subj':0}
        pattern_des = {'type':np.array([0,1,2,2,4])}
        rdm_des = {'session':np.array([0,1,2,2,4,5,6,7])}
        rdms = rsa.rdm.RDMs(dissimilarities=dis,
                        pattern_descriptors=pattern_des,
                        dissimilarity_measure=mes,
                        descriptors=des,
                        rdm_descriptors=rdm_des)
        rdm_dict = rdms.to_dict()
        rdms_loaded = rsa.rdm.rdms_from_dict(rdm_dict)
        assert rdms_loaded.n_cond == rdms.n_cond
        assert np.all(rdms_loaded.pattern_descriptors['type'] == pattern_des['type'])
        assert np.all(rdms_loaded.rdm_descriptors['session'] == rdm_des['session'])
        assert rdms_loaded.descriptors['subj'] == 0

    def test_save_load(self):
        import io
        f = io.BytesIO() # Essentially a Mock file
        dis = np.zeros((8,10))
        mes = "Euclidean"
        des = {'subj':0}
        pattern_des = {'type':np.array([0,1,2,2,4])}
        rdm_des = {'session':np.array([0,1,2,2,4,5,6,7])}
        rdms = rsa.rdm.RDMs(dissimilarities=dis,
                        pattern_descriptors=pattern_des,
                        dissimilarity_measure=mes,
                        descriptors=des,
                        rdm_descriptors=rdm_des)
        rdms.save(f, file_type='hdf5')
        rdms_loaded = rsa.rdm.load_rdm(f, file_type='hdf5')
        assert rdms_loaded.n_cond == rdms.n_cond
        assert np.all(rdms_loaded.pattern_descriptors['type'] == pattern_des['type'])
        assert np.all(rdms_loaded.rdm_descriptors['session'] == rdm_des['session'])
        assert rdms_loaded.descriptors['subj'] == 0
        


if __name__ == '__main__':
    unittest.main()  
