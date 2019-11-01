#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_indicator 

@author: jdiedrichsen
"""

import unittest 
import pyrsa.util as rsu
import numpy as np 


class TestIndicator(unittest.TestCase): 
    
    def test_identity(self):
        a = np.array(range(0,5))
        a = np.concatenate((a,a))
        X = rsu.indicator.identity(a)
        n_row,n_col = X.shape
        self.assertEqual(n_row,10)
        self.assertEqual(n_col,5)
        self.assertEqual(X[0,0],1.0)

    def test_identity_pos(self):
        a = np.array(range(0,5))
        a = np.concatenate((a,a))
        X = rsu.indicator.identity(a, positive=True)
        n_row,n_col = X.shape
        self.assertEqual(n_row,10)
        self.assertEqual(n_col,4)
        self.assertEqual(X[0,0],0.0)

    def test_allpairs(self):
        a = np.array(range(0,5))
        X = rsu.indicator.allpairs(a)
        n_row,n_col = X.shape
        self.assertEqual(n_row,10)
        self.assertEqual(n_col,5)
        self.assertEqual(X[0,0],0.0)

if __name__ == '__main__':
    unittest.main()
