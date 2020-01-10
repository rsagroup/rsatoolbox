#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 10:00:26 2020

@author: heiko
"""

import unittest
import numpy as np


class test_bootstrap(unittest.TestCase):
    """ bootstrap tests
    """
    def test_bootstrap_sample(self):
        from pyrsa.inference import bootstrap_sample
        from pyrsa.rdm import RDMs
        rdms = RDMs(np.random.rand(11,10))  # 11 5x5 rdms
        rdm_sample = bootstrap_sample(rdms)
        assert rdm_sample.n_cond == 5
        assert rdm_sample.n_rdm == 11

    def test_bootstrap_sample_descriptors(self):
        from pyrsa.inference import bootstrap_sample
        from pyrsa.rdm import RDMs
        dis = np.random.rand(11,10)  # 11 5x5 rdms
        mes = "Euclidean"
        des = {'subj':0}
        rdm_des = {'session':np.array([1,1,2,2,4,5,6,7,7,7,7])}
        pattern_des = {'type':np.array([0,1,2,2,4])}
        rdms = RDMs(dissimilarities=dis,
                    rdm_descriptors=rdm_des,
                    pattern_descriptors=pattern_des,
                    dissimilarity_measure=mes,
                    descriptors=des)
        rdm_sample = bootstrap_sample(rdms,'session','type')
        
    def test_bootstrap_sample_rdm(self):
        from pyrsa.inference import bootstrap_sample_rdm
        from pyrsa.rdm import RDMs
        rdms = RDMs(np.random.rand(11,10))  # 11 5x5 rdms
        rdm_sample = bootstrap_sample_rdm(rdms)
        assert rdm_sample.n_cond == 5
        assert rdm_sample.n_rdm == 11

    def test_bootstrap_sample_rdm_descriptors(self):
        from pyrsa.inference import bootstrap_sample_rdm
        from pyrsa.rdm import RDMs
        dis = np.random.rand(11,10)  # 11 5x5 rdms
        mes = "Euclidean"
        des = {'subj':0}
        rdm_des = {'session':np.array([0,1,2,2,4,5,6,7,7,7,7])}
        pattern_des = {'type':np.array([0,1,2,2,4])}
        rdms = RDMs(dissimilarities=dis,
                    rdm_descriptors=rdm_des,
                    pattern_descriptors=pattern_des,
                    dissimilarity_measure=mes,
                    descriptors=des)
        rdm_sample = bootstrap_sample_rdm(rdms,'session')

    def test_bootstrap_sample_pattern(self):
        from pyrsa.inference import bootstrap_sample_pattern
        from pyrsa.rdm import RDMs
        rdms = RDMs(np.random.rand(11,10))  # 11 5x5 rdms
        rdm_sample = bootstrap_sample_pattern(rdms)
        assert rdm_sample.n_cond == 5
        assert rdm_sample.n_rdm == 11

    def test_bootstrap_sample_pattern_descriptors(self):
        from pyrsa.inference import bootstrap_sample_pattern
        from pyrsa.rdm import RDMs
        dis = np.random.rand(11,10)  # 11 5x5 rdms
        mes = "Euclidean"
        des = {'subj':0}
        rdm_des = {'session':np.array([0,1,2,2,4,5,6,7,7,7,7])}
        pattern_des = {'type':np.array([0,1,2,2,4])}
        rdms = RDMs(dissimilarities=dis,
                    rdm_descriptors=rdm_des,
                    pattern_descriptors=pattern_des,
                    dissimilarity_measure=mes,
                    descriptors=des)
        rdm_sample = bootstrap_sample_pattern(rdms,'type')
