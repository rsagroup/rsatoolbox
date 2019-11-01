#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_data
Test for Dataset class 
@author: baihan, jdiedrichsen
"""

import unittest 
import pyrsa.data as rsd
import numpy as np 

class TestData(unittest.TestCase): 
    
    def test_dataset_simple_init(self):
        measurements = np.zeros((10,5))
        data = rsd.Dataset(measurements)
        self.assertEqual(data.n_obs,10)
        self.assertEqual(data.n_channel,5)

    def test_dataset_full_init(self):
        measurements = np.zeros((10,5))
        des = {"session":0,"subj":0}
        obs_des = {["conds_"+str(x) for x in np.arange(10)]}
        chn_des = {["roi_"+str(x) for x in np.arange(5)]}
        data = rsd.Dataset(measurements,des,obs_des,chn_des)
        self.assertEqual(data.n_obs,10)
        self.assertEqual(data.n_channel,5)
        self.assertEqual(data.des,des)
        self.assertEqual(data.obs_des,obs_des)
        self.assertEqual(data.chn_des,chn_des)

if __name__ == '__main__':
    unittest.main()        