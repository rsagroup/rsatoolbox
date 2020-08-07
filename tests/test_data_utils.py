#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_data_utils
Test for Dataset utils
@author: alex
"""

import unittest
from pyrsa.util import data_utils as du
import numpy as np

class TestGetUniqueUnsorted(unittest.TestCase):
    def setUp(self):
        self.full_str = "slight not what is near through aiming at what is far"
        self.unique_str = "slight not what is near through aiming at far"
        self.full_ints = np.array([99, 4, 66, 4, 33, 99])
        self.unique_ints = np.array([99, 4, 66, 33])
        
    def test_get_unique_unsorted_str(self):
        self.array = np.array([self.full_str.split(' ')])
        self.unique_unsorted = du.get_unique_unsorted(self.array)
        assert np.all(self.unique_unsorted == self.unique_str.split(' '))
        
    def test_get_unique_unsorted_ints(self):
        self.array = np.array([self.full_ints])
        self.unique_unsorted = du.get_unique_unsorted(self.array)
        assert np.all(self.unique_unsorted == self.unique_ints)

if __name__ == '__main__':
    unittest.main()
