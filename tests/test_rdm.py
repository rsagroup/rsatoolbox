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
        
    def test_calc_correlation(self):
        rdm = rsr.calc_rdm_correlation(self.test_data, descriptor = 'conds')
        assert rdm.n_cond == 6
        
    def test_calc_mahalanobis(self):
        rdm = rsr.calc_rdm(self.test_data, descriptor = 'conds', method = 'mahalanobis')
        assert rdm.n_cond == 6
        
    def test_calc_crossnobis(self):
        rdm = rsr.calc_rdm_crossnobis(self.test_data, descriptor = 'conds', cv_descriptor = 'fold')
        assert rdm.n_cond == 6

if __name__ == '__main__':
    unittest.main()  
