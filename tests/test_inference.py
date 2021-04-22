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
        from rsatoolbox.inference import bootstrap_sample
        from rsatoolbox.rdm import RDMs
        rdms = RDMs(np.random.rand(11, 10))  # 11 5x5 rdms
        rdm_sample = bootstrap_sample(rdms)
        assert rdm_sample[0].n_cond == 5
        assert rdm_sample[0].n_rdm == 11

    def test_bootstrap_sample_descriptors(self):
        from rsatoolbox.inference import bootstrap_sample
        from rsatoolbox.rdm import RDMs
        dis = np.random.rand(11, 10)  # 11 5x5 rdms
        mes = "Euclidean"
        des = {'subj': 0}
        rdm_des = {'session': np.array([1, 1, 2, 2, 4, 5, 6, 7, 7, 7, 7])}
        pattern_des = {'type': np.array([0, 1, 2, 3, 4])}
        rdms = RDMs(dissimilarities=dis,
                    rdm_descriptors=rdm_des,
                    pattern_descriptors=pattern_des,
                    dissimilarity_measure=mes,
                    descriptors=des)
        rdm_sample = bootstrap_sample(rdms, 'session', 'type')
        assert rdm_sample[0].n_cond == 5

    def test_bootstrap_sample_rdm(self):
        from rsatoolbox.inference import bootstrap_sample_rdm
        from rsatoolbox.rdm import RDMs
        rdms = RDMs(np.random.rand(11, 10))  # 11 5x5 rdms
        rdm_sample = bootstrap_sample_rdm(rdms)
        assert rdm_sample[0].n_cond == 5
        assert rdm_sample[0].n_rdm == 11

    def test_bootstrap_sample_rdm_descriptors(self):
        from rsatoolbox.inference import bootstrap_sample_rdm
        from rsatoolbox.rdm import RDMs
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
        assert rdm_sample[0].n_cond == 5

    def test_bootstrap_sample_pattern(self):
        from rsatoolbox.inference import bootstrap_sample_pattern
        from rsatoolbox.rdm import RDMs
        rdms = RDMs(np.random.rand(11, 10))  # 11 5x5 rdms
        rdm_sample = bootstrap_sample_pattern(rdms)
        assert rdm_sample[0].n_cond == 5
        assert rdm_sample[0].n_rdm == 11

    def test_bootstrap_sample_pattern_descriptors(self):
        from rsatoolbox.inference import bootstrap_sample_pattern
        from rsatoolbox.rdm import RDMs
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
        rdm_sample = bootstrap_sample_pattern(rdms)
        assert rdm_sample[0].n_cond == 5


class TestEvaluation(unittest.TestCase):
    """ evaluation tests
    """

    def test_eval_fixed(self):
        from rsatoolbox.inference import eval_fixed
        from rsatoolbox.rdm import RDMs
        from rsatoolbox.model import ModelFixed
        rdms = RDMs(np.random.rand(11, 10))  # 11 5x5 rdms
        m = ModelFixed('test', rdms.get_vectors()[0])
        eval_fixed(m, rdms)

    def test_eval_bootstrap(self):
        from rsatoolbox.inference import eval_bootstrap
        from rsatoolbox.rdm import RDMs
        from rsatoolbox.model import ModelFixed
        rdms = RDMs(np.random.rand(11, 10))  # 11 5x5 rdms
        m = ModelFixed('test', rdms.get_vectors()[0])
        eval_bootstrap(m, rdms, N=10)

    def test_eval_bootstrap_pattern(self):
        from rsatoolbox.inference import eval_bootstrap_pattern
        from rsatoolbox.rdm import RDMs
        from rsatoolbox.model import ModelFixed
        rdms = RDMs(np.random.rand(11, 10))  # 11 5x5 rdms
        m = ModelFixed('test', rdms.get_vectors()[0])
        eval_bootstrap_pattern(m, rdms, N=10)

    def test_eval_bootstrap_rdm(self):
        from rsatoolbox.inference import eval_bootstrap_rdm
        from rsatoolbox.rdm import RDMs
        from rsatoolbox.model import ModelFixed
        rdms = RDMs(np.random.rand(11, 10))  # 11 5x5 rdms
        m = ModelFixed('test', rdms.get_vectors()[0])
        eval_bootstrap_rdm(m, rdms, N=10)
        eval_bootstrap_rdm(m, rdms, N=10, boot_noise_ceil=True)

    def test_bootstrap_testset(self):
        from rsatoolbox.inference import bootstrap_testset
        from rsatoolbox.rdm import RDMs
        from rsatoolbox.model import ModelFixed
        rdms = RDMs(np.random.rand(11, 10))  # 11 5x5 rdms
        m = ModelFixed('test', rdms.get_vectors()[0])
        bootstrap_testset(m, rdms, method='cosine', fitter=None, N=100,
                          pattern_descriptor=None, rdm_descriptor=None)

    def test_bootstrap_testset_pattern(self):
        from rsatoolbox.inference import bootstrap_testset_pattern
        from rsatoolbox.rdm import RDMs
        from rsatoolbox.model import ModelFixed
        rdms = RDMs(np.random.rand(11, 10))  # 11 5x5 rdms
        m = ModelFixed('test', rdms.get_vectors()[0])
        _, _ = bootstrap_testset_pattern(
            m, rdms,
            method='cosine', fitter=None, N=100, pattern_descriptor=None)

    def test_bootstrap_testset_rdm(self):
        from rsatoolbox.inference import bootstrap_testset_rdm
        from rsatoolbox.rdm import RDMs
        from rsatoolbox.model import ModelFixed
        rdms = RDMs(np.random.rand(11, 10))  # 11 5x5 rdms
        m = ModelFixed('test', rdms.get_vectors()[0])
        _, _ = bootstrap_testset_rdm(
            m, rdms,
            method='cosine', fitter=None, N=100, rdm_descriptor=None)


class TestEvaluationLists(unittest.TestCase):
    """ evaluation tests
    """

    def test_eval_fixed(self):
        from rsatoolbox.inference import eval_fixed
        from rsatoolbox.rdm import RDMs
        from rsatoolbox.model import ModelFixed
        rdms = RDMs(np.random.rand(11, 10))  # 11 5x5 rdms
        m = ModelFixed('test', rdms.get_vectors()[0])
        m2 = ModelFixed('test2', rdms.get_vectors()[1])
        result = eval_fixed([m, m2], rdms)
        assert result.n_model == 2
        assert result.evaluations.shape[1] == 2

    def test_eval_bootstrap(self):
        from rsatoolbox.inference import eval_bootstrap
        from rsatoolbox.rdm import RDMs
        from rsatoolbox.model import ModelFixed
        rdms = RDMs(np.random.rand(11, 10))  # 11 5x5 rdms
        m = ModelFixed('test', rdms.get_vectors()[0])
        m2 = ModelFixed('test2', rdms.get_vectors()[1])
        result = eval_bootstrap([m, m2], rdms, N=10)
        assert result.evaluations.shape[1] == 2
        assert result.evaluations.shape[0] == 10

    def test_eval_bootstrap_pattern(self):
        from rsatoolbox.inference import eval_bootstrap_pattern
        from rsatoolbox.rdm import RDMs
        from rsatoolbox.model import ModelFixed
        rdms = RDMs(np.random.rand(11, 10))  # 11 5x5 rdms
        m = ModelFixed('test', rdms.get_vectors()[0])
        m2 = ModelFixed('test2', rdms.get_vectors()[1])
        value = eval_bootstrap_pattern([m, m2], rdms, N=10)

    def test_eval_bootstrap_rdm(self):
        from rsatoolbox.inference import eval_bootstrap_rdm
        from rsatoolbox.rdm import RDMs
        from rsatoolbox.model import ModelFixed
        rdms = RDMs(np.random.rand(11, 10))  # 11 5x5 rdms
        m = ModelFixed('test', rdms.get_vectors()[0])
        m2 = ModelFixed('test2', rdms.get_vectors()[1])
        value = eval_bootstrap_rdm([m, m2], rdms, N=10)

    def test_bootstrap_testset(self):
        from rsatoolbox.inference import bootstrap_testset
        from rsatoolbox.rdm import RDMs
        from rsatoolbox.model import ModelFixed
        rdms = RDMs(np.random.rand(11, 10))  # 11 5x5 rdms
        m = ModelFixed('test', rdms.get_vectors()[0])
        m2 = ModelFixed('test2', rdms.get_vectors()[1])
        bootstrap_testset([m, m2], rdms, method='cosine', fitter=None, N=100,
                          pattern_descriptor=None, rdm_descriptor=None)

    def test_bootstrap_testset_pattern(self):
        from rsatoolbox.inference import bootstrap_testset_pattern
        from rsatoolbox.rdm import RDMs
        from rsatoolbox.model import ModelFixed
        rdms = RDMs(np.random.rand(11, 10))  # 11 5x5 rdms
        m = ModelFixed('test', rdms.get_vectors()[0])
        m2 = ModelFixed('test2', rdms.get_vectors()[1])
        evaluations, n_cond = bootstrap_testset_pattern(
            [m, m2], rdms,
            method='cosine', fitter=None, N=100, pattern_descriptor=None)

    def test_bootstrap_testset_rdm(self):
        from rsatoolbox.inference import bootstrap_testset_rdm
        from rsatoolbox.rdm import RDMs
        from rsatoolbox.model import ModelFixed
        rdms = RDMs(np.random.rand(11, 10))  # 11 5x5 rdms
        m = ModelFixed('test', rdms.get_vectors()[0])
        m2 = ModelFixed('test2', rdms.get_vectors()[1])
        evaluations, n_rdms = bootstrap_testset_rdm(
            [m, m2], rdms,
            method='cosine', fitter=None, N=100, rdm_descriptor=None)


class TestSaveLoad(unittest.TestCase):
    def test_model_dict(self):
        from rsatoolbox.model import model_from_dict
        from rsatoolbox.model import ModelFixed
        m = ModelFixed('test1', np.random.rand(10))
        model_dict = m.to_dict()
        model_loaded = model_from_dict(model_dict)
        assert m.name == model_loaded.name
        assert np.all(m.rdm_obj.dissimilarities
                      == model_loaded.rdm_obj.dissimilarities)

        from rsatoolbox.model import ModelInterpolate
        m = ModelInterpolate('test1', np.random.rand(10))
        model_dict = m.to_dict()
        model_loaded = model_from_dict(model_dict)
        assert m.name == model_loaded.name
        assert np.all(m.rdm_obj.dissimilarities
                      == model_loaded.rdm_obj.dissimilarities)

        from rsatoolbox.model import ModelSelect
        m = ModelSelect('test1', np.random.rand(10))
        model_dict = m.to_dict()
        model_loaded = model_from_dict(model_dict)
        assert m.name == model_loaded.name
        assert np.all(m.rdm_obj.dissimilarities
                      == model_loaded.rdm_obj.dissimilarities)

        from rsatoolbox.model import ModelWeighted
        m = ModelWeighted('test1', np.random.rand(10))
        model_dict = m.to_dict()
        model_loaded = model_from_dict(model_dict)
        assert m.name == model_loaded.name
        assert np.all(m.rdm_obj.dissimilarities
                      == model_loaded.rdm_obj.dissimilarities)

        from rsatoolbox.model import Model
        m = Model('test1')
        model_dict = m.to_dict()
        model_loaded = model_from_dict(model_dict)
        assert m.name == model_loaded.name

    def test_result_dict(self):
        from rsatoolbox.inference import Result
        from rsatoolbox.inference import result_from_dict
        from rsatoolbox.model import ModelFixed
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
        from rsatoolbox.rdm import RDMs
        from rsatoolbox.inference import Result
        from rsatoolbox.inference import load_results
        from rsatoolbox.model import ModelFixed
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
        from rsatoolbox.util.inference_util import pair_tests
        ps = pair_tests(self.evaluations)
        assert np.all(ps <= 1)
        assert np.all(ps >= 0)

    def test_t_tests(self):
        from rsatoolbox.util.inference_util import t_tests
        variances = np.ones(10)
        ps = t_tests(self.evaluations, variances)
        assert np.all(ps <= 1)
        assert np.all(ps >= 0)

    def test_t_scipy(self):
        from rsatoolbox.util.inference_util import t_tests
        from rsatoolbox.inference import eval_fixed
        from rsatoolbox.rdm import RDMs
        from rsatoolbox.model import ModelFixed
        import scipy.stats

        rdms = RDMs(np.random.rand(11, 10))  # 11 5x5 rdms
        m = ModelFixed('test', rdms.get_vectors()[0])
        m2 = ModelFixed('test', rdms.get_vectors()[2])
        value = eval_fixed([m, m2], rdms)
        ps = t_tests(value.evaluations, value.diff_var, dof=value.dof)
        scipy_t = scipy.stats.ttest_rel(value.evaluations[0, 0],
                                        value.evaluations[0, 1])
        self.assertAlmostEqual(scipy_t.pvalue, ps[0, 1])

    def test_t_test_0(self):
        from rsatoolbox.util.inference_util import t_test_0
        variances = np.ones(5)
        ps = t_test_0(self.evaluations, variances)
        assert np.all(ps <= 1)
        assert np.all(ps >= 0)

    def test_t_test_nc(self):
        from rsatoolbox.util.inference_util import t_test_nc
        variances = np.array([0.01, 0.1, 0.2, 0.1, 0.1, 0.3])
        ps = t_test_nc(self.evaluations, variances, 0.3)
        assert np.all(ps <= 1)
        assert np.all(ps >= 0)

    def test_ranksum_test(self):
        from rsatoolbox.util.inference_util import ranksum_pair_test
        ps = ranksum_pair_test(self.evaluations)
        assert np.all(ps <= 1)
        assert np.all(ps >= 0)

    def test_sign_test_0(self):
        from rsatoolbox.util.inference_util import ranksum_value_test
        ps = ranksum_value_test(self.evaluations)
        assert np.all(ps <= 1)
        assert np.all(ps >= 0)

    def test_sign_test_value(self):
        from rsatoolbox.util.inference_util import ranksum_value_test
        ps = ranksum_value_test(self.evaluations, 0.3)
        assert np.all(ps <= 1)
        assert np.all(ps >= 0)


class TestsDefaultK(unittest.TestCase):

    def test_default_k_rdm(self):
        from rsatoolbox.util.inference_util import default_k_rdm
        self.assertEqual(default_k_rdm(5), 2)
        self.assertEqual(default_k_rdm(11), 3)
        self.assertEqual(default_k_rdm(19), 4)
        self.assertEqual(default_k_rdm(100), 5)

    def test_default_k_pattern(self):
        from rsatoolbox.util.inference_util import default_k_pattern
        self.assertEqual(default_k_pattern(10), 2)
        self.assertEqual(default_k_pattern(20), 3)
        self.assertEqual(default_k_pattern(30), 4)
        self.assertEqual(default_k_pattern(100), 5)


class TestsExtractVar(unittest.TestCase):

    def test_extract_var_1D(self):
        from rsatoolbox.util.inference_util import extract_variances
        variance = np.var(np.random.randn(10, 100), 1)
        model_variances, diff_variances, nc_variances = \
            extract_variances(variance, True)
        self.assertEqual(model_variances.shape[0], 8)
        self.assertEqual(diff_variances.shape[0], 28)
        self.assertEqual(nc_variances.shape[0], 8)
        self.assertEqual(nc_variances.shape[1], 2)

        model_variances, diff_variances, nc_variances = \
            extract_variances(variance, False)
        self.assertEqual(model_variances.shape[0], 10)
        self.assertEqual(diff_variances.shape[0], 45)
        self.assertEqual(nc_variances.shape[0], 10)
        self.assertEqual(nc_variances.shape[1], 2)

    def test_extract_var_2D(self):
        from rsatoolbox.util.inference_util import extract_variances
        variance = np.cov(np.random.randn(10, 100))
        model_variances, diff_variances, nc_variances = \
            extract_variances(variance, True)
        self.assertEqual(model_variances.shape[0], 8)
        self.assertEqual(diff_variances.shape[0], 28)
        self.assertEqual(nc_variances.shape[0], 8)
        self.assertEqual(nc_variances.shape[1], 2)

        model_variances, diff_variances, nc_variances = \
            extract_variances(variance, False)
        self.assertEqual(model_variances.shape[0], 10)
        self.assertEqual(diff_variances.shape[0], 45)
        self.assertEqual(nc_variances.shape[0], 10)
        self.assertEqual(nc_variances.shape[1], 2)

    def test_extract_var_3D(self):
        from rsatoolbox.util.inference_util import extract_variances
        variance = np.cov(np.random.randn(10, 100))
        variance = np.repeat(np.expand_dims(variance, 0), 3, 0
                             ).reshape(3, 10, 10)
        model_variances, diff_variances, nc_variances = \
            extract_variances(variance, True)
        self.assertEqual(model_variances.shape[0], 8)
        self.assertEqual(diff_variances.shape[0], 28)
        self.assertEqual(nc_variances.shape[0], 8)
        self.assertEqual(nc_variances.shape[1], 2)

        model_variances, diff_variances, nc_variances = \
            extract_variances(variance, False)
        self.assertEqual(model_variances.shape[0], 10)
        self.assertEqual(diff_variances.shape[0], 45)
        self.assertEqual(nc_variances.shape[0], 10)
        self.assertEqual(nc_variances.shape[1], 2)
