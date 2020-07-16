#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 10:00:26 2020

@author: heiko
"""

import unittest
import numpy as np


class TestBootstrap(unittest.TestCase):
    """ bootstrap tests
    """

    def test_bootstrap_sample(self):
        from pyrsa.inference import bootstrap_sample
        from pyrsa.rdm import RDMs
        rdms = RDMs(np.random.rand(11, 10))  # 11 5x5 rdms
        rdm_sample = bootstrap_sample(rdms)
        assert rdm_sample[0].n_cond == 5
        assert rdm_sample[0].n_rdm == 11

    def test_bootstrap_sample_descriptors(self):
        from pyrsa.inference import bootstrap_sample
        from pyrsa.rdm import RDMs
        dis = np.random.rand(11, 10)  # 11 5x5 rdms
        mes = "Euclidean"
        des = {'subj': 0}
        rdm_des = {'session': np.array([1, 1, 2, 2, 4, 5, 6, 7, 7, 7, 7])}
        pattern_des = {'type': np.array([0, 1, 2, 2, 4])}
        rdms = RDMs(dissimilarities=dis,
                    rdm_descriptors=rdm_des,
                    pattern_descriptors=pattern_des,
                    dissimilarity_measure=mes,
                    descriptors=des)
        rdm_sample = bootstrap_sample(rdms, 'session', 'type')

    def test_bootstrap_sample_rdm(self):
        from pyrsa.inference import bootstrap_sample_rdm
        from pyrsa.rdm import RDMs
        rdms = RDMs(np.random.rand(11, 10))  # 11 5x5 rdms
        rdm_sample = bootstrap_sample_rdm(rdms)
        assert rdm_sample[0].n_cond == 5
        assert rdm_sample[0].n_rdm == 11

    def test_bootstrap_sample_rdm_descriptors(self):
        from pyrsa.inference import bootstrap_sample_rdm
        from pyrsa.rdm import RDMs
        dis = np.random.rand(11, 10)  # 11 5x5 rdms
        mes = "Euclidean"
        des = {'subj': 0}
        rdm_des = {'session': np.array([0, 1, 2, 2, 4, 5, 6, 7, 7, 7, 7])}
        pattern_des = {'type': np.array([0, 1, 2, 2, 4])}
        rdms = RDMs(dissimilarities=dis,
                    rdm_descriptors=rdm_des,
                    pattern_descriptors=pattern_des,
                    dissimilarity_measure=mes,
                    descriptors=des)
        rdm_sample = bootstrap_sample_rdm(rdms, 'session')

    def test_bootstrap_sample_pattern(self):
        from pyrsa.inference import bootstrap_sample_pattern
        from pyrsa.rdm import RDMs
        rdms = RDMs(np.random.rand(11, 10))  # 11 5x5 rdms
        rdm_sample = bootstrap_sample_pattern(rdms)
        assert rdm_sample[0].n_cond == 5
        assert rdm_sample[0].n_rdm == 11

    def test_bootstrap_sample_pattern_descriptors(self):
        from pyrsa.inference import bootstrap_sample_pattern
        from pyrsa.rdm import RDMs
        dis = np.random.rand(11, 10)  # 11 5x5 rdms
        mes = "Euclidean"
        des = {'subj': 0}
        rdm_des = {'session': np.array([0, 1, 2, 2, 4, 5, 6, 7, 7, 7, 7])}
        pattern_des = {'type': np.array([0, 1, 2, 2, 4])}
        rdms = RDMs(dissimilarities=dis,
                    rdm_descriptors=rdm_des,
                    pattern_descriptors=pattern_des,
                    dissimilarity_measure=mes,
                    descriptors=des)
        rdm_sample = bootstrap_sample_pattern(rdms, 'type')


class TestEvaluation(unittest.TestCase):
    """ evaluation tests
    """

    def test_eval_fixed(self):
        from pyrsa.inference import eval_fixed
        from pyrsa.rdm import RDMs
        from pyrsa.model import ModelFixed
        rdms = RDMs(np.random.rand(11, 10))  # 11 5x5 rdms
        m = ModelFixed('test', rdms.get_vectors()[0])
        value = eval_fixed(m, rdms)

    def test_eval_bootstrap(self):
        from pyrsa.inference import eval_bootstrap
        from pyrsa.rdm import RDMs
        from pyrsa.model import ModelFixed
        rdms = RDMs(np.random.rand(11, 10))  # 11 5x5 rdms
        m = ModelFixed('test', rdms.get_vectors()[0])
        value = eval_bootstrap(m, rdms, N=10)

    def test_eval_bootstrap_pattern(self):
        from pyrsa.inference import eval_bootstrap_pattern
        from pyrsa.rdm import RDMs
        from pyrsa.model import ModelFixed
        rdms = RDMs(np.random.rand(11, 10))  # 11 5x5 rdms
        m = ModelFixed('test', rdms.get_vectors()[0])
        value = eval_bootstrap_pattern(m, rdms, N=10)

    def test_eval_bootstrap_rdm(self):
        from pyrsa.inference import eval_bootstrap_rdm
        from pyrsa.rdm import RDMs
        from pyrsa.model import ModelFixed
        rdms = RDMs(np.random.rand(11, 10))  # 11 5x5 rdms
        m = ModelFixed('test', rdms.get_vectors()[0])
        value = eval_bootstrap_rdm(m, rdms, N=10)

    def test_bootstrap_testset(self):
        from pyrsa.inference import bootstrap_testset
        from pyrsa.rdm import RDMs
        from pyrsa.model import ModelFixed
        rdms = RDMs(np.random.rand(11, 10))  # 11 5x5 rdms
        m = ModelFixed('test', rdms.get_vectors()[0])
        bootstrap_testset(m, rdms, method='cosine', fitter=None, N=100,
                          pattern_descriptor=None, rdm_descriptor=None)

    def test_bootstrap_testset_pattern(self):
        from pyrsa.inference import bootstrap_testset_pattern
        from pyrsa.rdm import RDMs
        from pyrsa.model import ModelFixed
        rdms = RDMs(np.random.rand(11, 10))  # 11 5x5 rdms
        m = ModelFixed('test', rdms.get_vectors()[0])
        evaluations, n_cond = bootstrap_testset_pattern(m, rdms,
            method='cosine', fitter=None, N=100, pattern_descriptor=None)

    def test_bootstrap_testset_rdm(self):
        from pyrsa.inference import bootstrap_testset_rdm
        from pyrsa.rdm import RDMs
        from pyrsa.model import ModelFixed
        rdms = RDMs(np.random.rand(11, 10))  # 11 5x5 rdms
        m = ModelFixed('test', rdms.get_vectors()[0])
        evaluations, n_rdms = bootstrap_testset_rdm(m, rdms,
            method='cosine', fitter=None, N=100, rdm_descriptor=None)


class TestEvaluationLists(unittest.TestCase):
    """ evaluation tests
    """
    def test_eval_fixed(self):
        from pyrsa.inference import eval_fixed
        from pyrsa.rdm import RDMs
        from pyrsa.model import ModelFixed
        rdms = RDMs(np.random.rand(11, 10))  # 11 5x5 rdms
        m = ModelFixed('test', rdms.get_vectors()[0])
        m2 = ModelFixed('test2', rdms.get_vectors()[1])
        result = eval_fixed([m, m2], rdms)
        assert result.n_model == 2
        assert result.evaluations.shape[1] == 2

    def test_eval_bootstrap(self):
        from pyrsa.inference import eval_bootstrap
        from pyrsa.rdm import RDMs
        from pyrsa.model import ModelFixed
        rdms = RDMs(np.random.rand(11, 10))  # 11 5x5 rdms
        m = ModelFixed('test', rdms.get_vectors()[0])
        m2 = ModelFixed('test2', rdms.get_vectors()[1])
        result = eval_bootstrap([m, m2], rdms, N=10)
        assert result.evaluations.shape[1] == 2
        assert result.evaluations.shape[0] == 10

    def test_eval_bootstrap_pattern(self):
        from pyrsa.inference import eval_bootstrap_pattern
        from pyrsa.rdm import RDMs
        from pyrsa.model import ModelFixed
        rdms = RDMs(np.random.rand(11, 10))  # 11 5x5 rdms
        m = ModelFixed('test', rdms.get_vectors()[0])
        m2 = ModelFixed('test2', rdms.get_vectors()[1])
        value = eval_bootstrap_pattern([m, m2], rdms, N=10)

    def test_eval_bootstrap_rdm(self):
        from pyrsa.inference import eval_bootstrap_rdm
        from pyrsa.rdm import RDMs
        from pyrsa.model import ModelFixed
        rdms = RDMs(np.random.rand(11, 10))  # 11 5x5 rdms
        m = ModelFixed('test', rdms.get_vectors()[0])
        m2 = ModelFixed('test2', rdms.get_vectors()[1])
        value = eval_bootstrap_rdm([m, m2], rdms, N=10)

    def test_bootstrap_testset(self):
        from pyrsa.inference import bootstrap_testset
        from pyrsa.rdm import RDMs
        from pyrsa.model import ModelFixed
        rdms = RDMs(np.random.rand(11, 10))  # 11 5x5 rdms
        m = ModelFixed('test', rdms.get_vectors()[0])
        m2 = ModelFixed('test2', rdms.get_vectors()[1])
        bootstrap_testset([m, m2], rdms, method='cosine', fitter=None, N=100,
                          pattern_descriptor=None, rdm_descriptor=None)

    def test_bootstrap_testset_pattern(self):
        from pyrsa.inference import bootstrap_testset_pattern
        from pyrsa.rdm import RDMs
        from pyrsa.model import ModelFixed
        rdms = RDMs(np.random.rand(11, 10))  # 11 5x5 rdms
        m = ModelFixed('test', rdms.get_vectors()[0])
        m2 = ModelFixed('test2', rdms.get_vectors()[1])
        evaluations, n_cond = bootstrap_testset_pattern(
            [m, m2], rdms,
            method='cosine', fitter=None, N=100, pattern_descriptor=None)

    def test_bootstrap_testset_rdm(self):
        from pyrsa.inference import bootstrap_testset_rdm
        from pyrsa.rdm import RDMs
        from pyrsa.model import ModelFixed
        rdms = RDMs(np.random.rand(11, 10))  # 11 5x5 rdms
        m = ModelFixed('test', rdms.get_vectors()[0])
        m2 = ModelFixed('test2', rdms.get_vectors()[1])
        evaluations, n_rdms = bootstrap_testset_rdm(
            [m, m2], rdms,
            method='cosine', fitter=None, N=100, rdm_descriptor=None)


class TestSaveLoad(unittest.TestCase):
    def test_model_dict(self):
        from pyrsa.model import model_from_dict
        from pyrsa.model import ModelFixed
        m = ModelFixed('test1', np.random.rand(10))
        model_dict = m.to_dict()
        model_loaded = model_from_dict(model_dict)
        assert m.name == model_loaded.name
        assert np.all(m.rdm_obj.dissimilarities
                      == model_loaded.rdm_obj.dissimilarities)

        from pyrsa.model import ModelInterpolate
        m = ModelInterpolate('test1', np.random.rand(10))
        model_dict = m.to_dict()
        model_loaded = model_from_dict(model_dict)
        assert m.name == model_loaded.name
        assert np.all(m.rdm_obj.dissimilarities
                      == model_loaded.rdm_obj.dissimilarities)

        from pyrsa.model import ModelSelect
        m = ModelSelect('test1', np.random.rand(10))
        model_dict = m.to_dict()
        model_loaded = model_from_dict(model_dict)
        assert m.name == model_loaded.name
        assert np.all(m.rdm_obj.dissimilarities
                      == model_loaded.rdm_obj.dissimilarities)

        from pyrsa.model import ModelWeighted
        m = ModelWeighted('test1', np.random.rand(10))
        model_dict = m.to_dict()
        model_loaded = model_from_dict(model_dict)
        assert m.name == model_loaded.name
        assert np.all(m.rdm_obj.dissimilarities
                      == model_loaded.rdm_obj.dissimilarities)

        from pyrsa.model import Model
        m = Model('test1')
        model_dict = m.to_dict()
        model_loaded = model_from_dict(model_dict)
        assert m.name == model_loaded.name

    def test_result_dict(self):
        from pyrsa.inference import Result
        from pyrsa.inference import result_from_dict
        from pyrsa.model import ModelFixed
        m1 = ModelFixed('test1', np.random.rand(10))
        m2 = ModelFixed('test2', np.random.rand(10))
        models = [m1, m2]
        evaluations = np.random.rand(100, 2)
        method = 'test_method'
        cv_method = 'test_cv_method'
        noise_ceiling = np.array([0.5, 0.2])
        res = Result(models, evaluations, method, cv_method, noise_ceiling)
        result_dict = res.to_dict()
        res_loaded = result_from_dict(result_dict)
        assert res_loaded.method == method
        assert res_loaded.cv_method == cv_method
        assert np.all(res_loaded.evaluations == evaluations)
        assert np.all(res_loaded.models[0].rdm == m1.rdm)

    def test_save_load_result(self):
        from pyrsa.rdm import RDMs
        from pyrsa.inference import Result
        from pyrsa.inference import load_results
        from pyrsa.model import ModelFixed
        import io
        rdm = RDMs(
            np.random.rand(10),
            pattern_descriptors={
                'test': ['test1', 'test1', 'test1', 'test3', 'test']})
        m1 = ModelFixed('test1', rdm)
        m2 = ModelFixed('test2', np.random.rand(10))
        models = [m1, m2]
        evaluations = np.random.rand(100, 2)
        method = 'test_method'
        cv_method = 'test_cv_method'
        noise_ceiling = np.array([0.5, 0.2])
        res = Result(models, evaluations, method, cv_method, noise_ceiling)
        f = io.BytesIO()  # Essentially a Mock file
        res.save(f, file_type='hdf5')
        res_loaded = load_results(f, file_type='hdf5')
        assert res_loaded.method == method
        assert res_loaded.cv_method == cv_method
        assert np.all(res_loaded.evaluations == evaluations)


class TestsPairTests(unittest.TestCase):

    def setUp(self):
        self.evaluations = np.random.rand(100, 5, 10)

    def test_pair_tests(self):
        from pyrsa.util.inference_util import pair_tests
        ps = pair_tests(self.evaluations)
        assert np.all(ps <= 1)
        assert np.all(ps >= 0)

    def test_t_tests(self):
        from pyrsa.util.inference_util import t_tests
        ps = t_tests(self.evaluations)
        assert np.all(ps <= 1)
        assert np.all(ps >= 0)
        variances = np.eye(5)
        ps = t_tests(self.evaluations, variances)
        assert np.all(ps <= 1)
        assert np.all(ps >= 0)
