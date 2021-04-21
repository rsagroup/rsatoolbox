#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 10:17:51 2020

@author: heiko
"""

import unittest
import numpy as np


class TestRdmUtils(unittest.TestCase): 
    
    def test_batch_to_vectors(self):
        from rsatoolbox.util.rdm_utils import batch_to_vectors
        dis = np.zeros((8,5,5))
        y, n_rdm, n_cond = batch_to_vectors(dis)
        assert y.shape[0] == 8
        assert y.shape[1] == 10
        assert n_rdm == 8
        assert n_cond == 5

    def test_batch_to_matrices(self):
        from rsatoolbox.util.rdm_utils import batch_to_matrices
        dis = np.zeros((8,5,5))
        y, n_rdm, n_cond = batch_to_matrices(dis)
        assert y.shape[0] == 8
        assert y.shape[1] == 5
        assert y.shape[2] == 5
        assert n_rdm == 8
        assert n_cond == 5
