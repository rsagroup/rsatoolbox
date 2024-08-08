#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_data_utils
Test for Dataset utils
@author: alex
"""

import unittest
import numpy as np
from rsatoolbox.util import data_utils as du
import rsatoolbox.data as rsd


class TestGetUniqueUnsorted(unittest.TestCase):
    def setUp(self):
        self.full_str = "slight not what is near through aiming at what is far"
        self.unique_str = "slight not what is near through aiming at far"
        self.full_ints = np.array([99, 4, 66, 4, 33, 99])
        self.unique_ints = np.array([99, 4, 66, 33])
        
        measurements = np.zeros((4, 5))
        des = {'session': 0, 'subj': 0}
        obs_des = {'conds': np.array(['cond_foo', 'cond_bar', 'cond_foo', 'cond_bar'])}
        chn_des = {'rois': np.array(['V1', 'V1', 'IT', 'IT', 'V4'])}
        self.data = rsd.Dataset(measurements=measurements,
                           descriptors=des,
                           obs_descriptors=obs_des,
                           channel_descriptors=chn_des
                           )
        
    def test_get_unique_unsorted_str(self):
        self.array = np.array([self.full_str.split(' ')])
        self.unique_unsorted = du.get_unique_unsorted(self.array)
        assert np.all(self.unique_unsorted == self.unique_str.split(' '))
        
    def test_get_unique_unsorted_ints(self):
        self.array = np.array([self.full_ints])
        self.unique_unsorted = du.get_unique_unsorted(self.array)
        assert np.all(self.unique_unsorted == self.unique_ints)
    
    def test_get_unique_unsorted_ds(self):
        unique_values = du.get_unique_unsorted(self.data.obs_descriptors['conds'])
        assert np.all(np.array(['cond_foo', 'cond_bar']) == unique_values)
        
if __name__ == '__main__':
    unittest.main()
