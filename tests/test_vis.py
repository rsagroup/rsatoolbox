#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_data
Test for visualization class 
@author: baihan
"""

import unittest 
import pyrsa.vis as rsv
import pyrsa.rdm as rsr
import numpy as np 

class TestVIS(unittest.TestCase): 
    
    def test_rdm2d_mds(self):
        dis = np.zeros((8,10))
        mes = "Euclidean"
        des = {'session':0, 'subj':0}
        rdms = rsr.RDMs(dissimilarities=dis,
                        dissimilarity_measure=mes,
                        descriptors=des)
        mds_emb = rsv.mds(rdms)
        self.assertEqual(mds_emb.shape,(5, 2))

if __name__ == '__main__':
    unittest.main()  