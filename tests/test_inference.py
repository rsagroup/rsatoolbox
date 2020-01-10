#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 10:00:26 2020

@author: heiko
"""

import unittest
import numpy as np


class test_bootstrap(unittest.TestCase):
    """ Tests for the fixed model class
    """
    def test_bootstrap_sample(self):
        from pyrsa.inference import bootstrap_sample
        from pyrsa.rdm import RDMs
        rdms = RDMs(np.random.rand(11,10))  # 11 5x5 rdms
        rdm_sample = bootstrap_sample(rdms)
