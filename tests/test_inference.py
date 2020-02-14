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

    def test_eval_bootstrap(self):
        from pyrsa.inference import eval_bootstrap
        from pyrsa.rdm import RDMs
        from pyrsa.model import ModelFixed
        rdms = RDMs(np.random.rand(11,10))  # 11 5x5 rdms
        m = ModelFixed('test', rdms.get_vectors()[0])
        value = eval_bootstrap(m, rdms, N=10)

    def test_eval_bootstrap_pattern(self):
        from pyrsa.inference import eval_bootstrap_pattern
        from pyrsa.rdm import RDMs
        from pyrsa.model import ModelFixed
        rdms = RDMs(np.random.rand(11,10))  # 11 5x5 rdms
        m = ModelFixed('test', rdms.get_vectors()[0])
        value = eval_bootstrap_pattern(m, rdms, N=10)

    def test_eval_bootstrap_rdm(self):
        from pyrsa.inference import eval_bootstrap_rdm
        from pyrsa.rdm import RDMs
        from pyrsa.model import ModelFixed
        rdms = RDMs(np.random.rand(11,10))  # 11 5x5 rdms
        m = ModelFixed('test', rdms.get_vectors()[0])
        value = eval_bootstrap_rdm(m, rdms, N=10)

    def test_bootstrap_testset(self):
        from pyrsa.inference import bootstrap_testset
        from pyrsa.rdm import RDMs
        from pyrsa.model import ModelFixed
        rdms = RDMs(np.random.rand(11,10))  # 11 5x5 rdms
        m = ModelFixed('test', rdms.get_vectors()[0])
        bootstrap_testset(m, rdms, method='cosine', fitter=None, N=100,
                          pattern_descriptor=None, rdm_descriptor=None)

    def test_bootstrap_testset_pattern(self):
        from pyrsa.inference import bootstrap_testset_pattern
        from pyrsa.rdm import RDMs
        from pyrsa.model import ModelFixed
        rdms = RDMs(np.random.rand(11,10))  # 11 5x5 rdms
        m = ModelFixed('test', rdms.get_vectors()[0])
        evaluations, n_cond = bootstrap_testset_pattern(m, rdms,
            method='cosine', fitter=None, N=100, pattern_descriptor=None)

    def test_bootstrap_testset_rdm(self):
        from pyrsa.inference import bootstrap_testset_rdm
        from pyrsa.rdm import RDMs
        from pyrsa.model import ModelFixed
        rdms = RDMs(np.random.rand(11,10))  # 11 5x5 rdms
        m = ModelFixed('test', rdms.get_vectors()[0])
        evaluations, n_rdms = bootstrap_testset_rdm(m, rdms,
            method='cosine', fitter=None, N=100, rdm_descriptor=None)

class test_evaluation_lists(unittest.TestCase):
    """ evaluation tests
    """
    def test_eval_fixed(self):
        from pyrsa.inference import eval_fixed
        from pyrsa.rdm import RDMs
        from pyrsa.model import ModelFixed
        rdms = RDMs(np.random.rand(11,10))  # 11 5x5 rdms
        m = ModelFixed('test', rdms.get_vectors()[0])
        m2 = ModelFixed('test2', rdms.get_vectors()[1])
        value = eval_fixed([m,m2], rdms)
        assert len(value) == 2

    def test_eval_bootstrap(self):
        from pyrsa.inference import eval_bootstrap
        from pyrsa.rdm import RDMs
        from pyrsa.model import ModelFixed
        rdms = RDMs(np.random.rand(11,10))  # 11 5x5 rdms
        m = ModelFixed('test', rdms.get_vectors()[0])
        m2 = ModelFixed('test2', rdms.get_vectors()[1])
        value = eval_bootstrap([m, m2], rdms, N=10)
        assert value.shape[1] == 2 and value.shape[0] == 10

    def test_eval_bootstrap_pattern(self):
        from pyrsa.inference import eval_bootstrap_pattern
        from pyrsa.rdm import RDMs
        from pyrsa.model import ModelFixed
        rdms = RDMs(np.random.rand(11,10))  # 11 5x5 rdms
        m = ModelFixed('test', rdms.get_vectors()[0])
        m2 = ModelFixed('test2', rdms.get_vectors()[1])
        value = eval_bootstrap_pattern([m, m2], rdms, N=10)

    def test_eval_bootstrap_rdm(self):
        from pyrsa.inference import eval_bootstrap_rdm
        from pyrsa.rdm import RDMs
        from pyrsa.model import ModelFixed
        rdms = RDMs(np.random.rand(11,10))  # 11 5x5 rdms
        m = ModelFixed('test', rdms.get_vectors()[0])
        m2 = ModelFixed('test2', rdms.get_vectors()[1])
        value = eval_bootstrap_rdm([m, m2], rdms, N=10)

    def test_bootstrap_testset(self):
        from pyrsa.inference import bootstrap_testset
        from pyrsa.rdm import RDMs
        from pyrsa.model import ModelFixed
        rdms = RDMs(np.random.rand(11,10))  # 11 5x5 rdms
        m = ModelFixed('test', rdms.get_vectors()[0])
        m2 = ModelFixed('test2', rdms.get_vectors()[1])
        bootstrap_testset([m, m2], rdms, method='cosine', fitter=None, N=100,
                          pattern_descriptor=None, rdm_descriptor=None)

    def test_bootstrap_testset_pattern(self):
        from pyrsa.inference import bootstrap_testset_pattern
        from pyrsa.rdm import RDMs
        from pyrsa.model import ModelFixed
        rdms = RDMs(np.random.rand(11,10))  # 11 5x5 rdms
        m = ModelFixed('test', rdms.get_vectors()[0])
        m2 = ModelFixed('test2', rdms.get_vectors()[1])
        evaluations, n_cond = bootstrap_testset_pattern([m, m2], rdms,
            method='cosine', fitter=None, N=100, pattern_descriptor=None)

    def test_bootstrap_testset_rdm(self):
        from pyrsa.inference import bootstrap_testset_rdm
        from pyrsa.rdm import RDMs
        from pyrsa.model import ModelFixed
        rdms = RDMs(np.random.rand(11,10))  # 11 5x5 rdms
        m = ModelFixed('test', rdms.get_vectors()[0])
        m2 = ModelFixed('test2', rdms.get_vectors()[1])
        evaluations, n_rdms = bootstrap_testset_rdm([m, m2], rdms,
            method='cosine', fitter=None, N=100, rdm_descriptor=None)
