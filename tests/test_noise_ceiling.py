#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 16:59:33 2020

@author: heiko
"""

import unittest
import numpy as np


class test_noise_ceiling(unittest.TestCase):
    def test_basic_noise_ceiling(self):
        from pyrsa.inference import noise_ceiling
        from pyrsa.inference import sets_k_fold_rdm
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
        train_sets, test_sets = sets_k_fold_rdm(rdms, k_rdm=3, random=False)
        noise_min, noise_max = noise_ceiling(train_sets, test_sets,
                                             method='cosine')
