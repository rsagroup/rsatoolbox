#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_data
Test for RDM class 
@author: baihan
"""

import unittest 
import pyrsa.rdm as rsr
import numpy as np 


class TestRDM(unittest.TestCase): 
    
    def test_rdm3d_init(self):
        dis = np.zeros((8,5,5))
        mes = "Euclidean"
        des = {'session':0, 'subj':0}
        rdms = rsr.RDMs(dissimilarities=dis,
                        dissimilarity_measure=mes,
                        descriptors=des)
        self.assertEqual(rdms.n_rdm,8)
        self.assertEqual(rdms.n_cond,4)

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
        self.assertEqual(v_rdms.shape[1],15)

    def test_rdm2d_get_vectors(self):
        dis = np.zeros((8,10))
        mes = "Euclidean"
        des = {'session':0, 'subj':0}
        rdms = rsr.RDMs(dissimilarities=dis,
                        dissimilarity_measure=mes,
                        descriptors=des)
        v_rdms = rdms.get_vectors()
        self.assertEqual(v_rdms.shape[0],8)
        self.assertEqual(v_rdms.shape[1],15)

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

if __name__ == '__main__':
    unittest.main()  
