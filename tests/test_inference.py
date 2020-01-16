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
        assert rdm_sample[0].n_cond == 5
        assert rdm_sample[0].n_rdm == 11

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
        assert rdm_sample[0].n_cond == 5
        assert rdm_sample[0].n_rdm == 11

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
        assert rdm_sample[0].n_cond == 5
        assert rdm_sample[0].n_rdm == 11

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

class test_evaluation(unittest.TestCase):
    """ evaluation tests
    """
    def test_eval_fixed(self):
        from pyrsa.inference import eval_fixed
        from pyrsa.rdm import RDMs
        from pyrsa.model import ModelFixed
        rdms = RDMs(np.random.rand(11,10))  # 11 5x5 rdms
        m = ModelFixed('test', rdms.get_vectors()[0])
        value = eval_fixed(m, rdms)


class test_crossval(unittest.TestCase):
    """ crossvalidation tests
    """
    def test_crossval(self):
        from pyrsa.inference import crossval
        from pyrsa.rdm import RDMs
        from pyrsa.model import ModelFixed
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
        m = ModelFixed('test', rdms[0])
        train_set = [(rdms.subset_pattern('type', [0,1]), np.array([0,1])),
                     (rdms.subset_pattern('type', [0,4]), np.array([0,4])),
                     ]
        test_set = [(rdms.subset_pattern('type', [2,4]), np.array([2,4])),
                    (rdms.subset_pattern('type', [1,2]), np.array([1,2])),
                    ]
        crossval(m, train_set, test_set, pattern_descriptor='type')

    def test_leave_one_out_pattern(self):
        from pyrsa.inference.evaluate import sets_leave_one_out_pattern
        import pyrsa.rdm as rsr
        dis = np.zeros((8,10))
        mes = "Euclidean"
        des = {'subj':0}
        rdm_des = {'session':np.array([0,1,2,2,4,5,6,7])}
        pattern_des = {'category':np.array([0,1,2,2,3])}
        rdms = rsr.RDMs(dissimilarities=dis,
                        rdm_descriptors=rdm_des,
                        dissimilarity_measure=mes,
                        pattern_descriptors=pattern_des,
                        descriptors=des)
        train_set, test_set = sets_leave_one_out_pattern(rdms)
        for i_test in test_set:
            i_test[0].n_cond == 1
