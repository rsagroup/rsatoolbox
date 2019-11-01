#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_data
Test for Dataset class 
@author: jdiedrichsen
"""

import unittest 
import pyrsa.data as rsd
import numpy as np 

class TestData(unittest.TestCase): 
    
    def test_dataset_init(self):
        A = np.zeros((10,5))
        data = rsd.Dataset(A)
        self.assertEqual(data.n_obs,10)
        self.assertEqual(data.n_channel,5)
        self.assertEqual(data.n_set,1)

if __name__ == '__main__':
    unittest.main()        