#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_data
Test for Dataset class 
@author: jdiedrichsen
"""

import unittest 
import pyrsa as rsa
import numpy as np 

class TestSimulation(unittest.TestCase): 
    
    def test_dataset2d(self):
        A = np.zeros((10,5))
        data = rsa.data.Dataset(A)
        self.assertEqual(data.n_obs,10)
        self.assertEqual(data.n_channel,5)
        self.assertEqual(data.n_set,1)

    def test_dataset3d(self):
        A = np.zeros((3,10,5))
        data = rsa.data.Dataset(A)
        self.assertEqual(data.n_obs,10)
        self.assertEqual(data.n_channel,5)
        self.assertEqual(data.n_set,3)

if __name__ == '__main__':
    unittest.main()        