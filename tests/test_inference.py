#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 10:00:26 2020

@author: heiko
"""

import unittest
import numpy as np
from rsatoolbox.rdm import RDMs


class TestBootstrap(unittest.TestCase):
    """ bootstrap tests
    """

    def setUp(self) -> None:
        self.rng = np.random.default_rng(0)
        dis = self.rng.random((11, 10))  # 11 5x5 rdms
        mes = "Euclidean"
        des = {'subj': 0}
        rdm_des = {'session': np.array([1, 1, 2, 2, 4, 5, 6, 7, 7, 7, 7])}
        pattern_des = {'type': np.array([0, 1, 2, 3, 4])}
        self.rdms = RDMs(
            dissimilarities=dis,
            rdm_descriptors=rdm_des,
            pattern_descriptors=pattern_des,
            dissimilarity_measure=mes,
            descriptors=des)
        return super().setUp()

    def test_bootstrap_sample(self):
        from rsatoolbox.inference import bootstrap_sample
        rdm_sample = bootstrap_sample(self.rdms)
        assert rdm_sample[0].n_cond == 5
        assert rdm_sample[0].n_rdm == 11

    def test_bootstrap_sample_descriptors(self):
        from rsatoolbox.inference import bootstrap_sample
        rdm_sample = bootstrap_sample(self.rdms, 'session', 'type')
        assert rdm_sample[0].n_cond == 5

    def test_bootstrap_sample_rdm(self):
        from rsatoolbox.inference import bootstrap_sample_rdm
        rdm_sample = bootstrap_sample_rdm(self.rdms)
        assert rdm_sample[0].n_cond == 5
        assert rdm_sample[0].n_rdm == 11

    def test_bootstrap_sample_rdm_descriptors(self):
        from rsatoolbox.inference import bootstrap_sample_rdm
        rdm_sample = bootstrap_sample_rdm(self.rdms, 'session')
        assert rdm_sample[0].n_cond == 5

    def test_bootstrap_sample_pattern(self):
        from rsatoolbox.inference import bootstrap_sample_pattern
        rdm_sample = bootstrap_sample_pattern(self.rdms)
        assert rdm_sample[0].n_cond == 5
        assert rdm_sample[0].n_rdm == 11

    def test_bootstrap_sample_pattern_descriptors(self):
        from rsatoolbox.inference import bootstrap_sample_pattern
        rdm_sample = bootstrap_sample_pattern(self.rdms, 'type')
        rdm_sample = bootstrap_sample_pattern(self.rdms)
        assert rdm_sample[0].n_cond == 5


class TestEvaluation(unittest.TestCase):
    """ evaluation tests
    """

    def setUp(self) -> None:
        from rsatoolbox.model import ModelFixed
        self.rng = np.random.default_rng(0)
        self.rdms = RDMs(self.rng.random((11, 10)))  # 11 5x5 rdms
        self.m = ModelFixed('test', self.rdms.get_vectors()[0])
        return super().setUp()

    def test_eval_fixed(self):
        from rsatoolbox.inference import eval_fixed
        eval_fixed(self.m, self.rdms)

    def test_eval_bootstrap(self):
        from rsatoolbox.inference import eval_bootstrap
        eval_bootstrap(self.m, self.rdms, N=10)

    def test_eval_bootstrap_pattern(self):
        from rsatoolbox.inference import eval_bootstrap_pattern
        eval_bootstrap_pattern(self.m, self.rdms, N=10)

    def test_eval_bootstrap_rdm(self):
        from rsatoolbox.inference import eval_bootstrap_rdm
        eval_bootstrap_rdm(self.m, self.rdms, N=10)
        eval_bootstrap_rdm(self.m, self.rdms, N=10, boot_noise_ceil=True)

    def test_bootstrap_testset(self):
        from rsatoolbox.inference import bootstrap_testset
        bootstrap_testset(self.m, self.rdms, method='cosine', fitter=None, N=100,
                          pattern_descriptor=None, rdm_descriptor=None)

    def test_bootstrap_testset_pattern(self):
        from rsatoolbox.inference import bootstrap_testset_pattern
        _, _ = bootstrap_testset_pattern(
            self.m, self.rdms,
            method='cosine', fitter=None, N=100, pattern_descriptor=None)

    def test_bootstrap_testset_rdm(self):
        from rsatoolbox.inference import bootstrap_testset_rdm
        _, _ = bootstrap_testset_rdm(
            self.m, self.rdms,
            method='cosine', fitter=None, N=100, rdm_descriptor=None)


class TestEvaluationLists(unittest.TestCase):
    """ evaluation tests
    """

    def setUp(self) -> None:
        from rsatoolbox.model import ModelFixed
        self.rng = np.random.default_rng(0)
        self.rdms = RDMs(self.rng.random((11, 10)))  # 11 5x5 rdms
        self.m = ModelFixed('test', self.rdms.get_vectors()[0])
        self.m2 = ModelFixed('test2', self.rdms.get_vectors()[1])
        return super().setUp()

    def test_eval_fixed(self):
        from rsatoolbox.inference import eval_fixed
        result = eval_fixed([self.m, self.m2], self.rdms)
        assert result.n_model == 2
        assert result.evaluations.shape[1] == 2

    def test_eval_bootstrap(self):
        from rsatoolbox.inference import eval_bootstrap
        result = eval_bootstrap([self.m, self.m2], self.rdms, N=10)
        assert result.evaluations.shape[1] == 2
        assert result.evaluations.shape[0] == 10

    def test_eval_bootstrap_pattern(self):
        from rsatoolbox.inference import eval_bootstrap_pattern
        _ = eval_bootstrap_pattern([self.m, self.m2], self.rdms, N=10)

    def test_eval_bootstrap_rdm(self):
        from rsatoolbox.inference import eval_bootstrap_rdm
        _ = eval_bootstrap_rdm([self.m, self.m2], self.rdms, N=10)

    def test_bootstrap_testset(self):
        from rsatoolbox.inference import bootstrap_testset
        bootstrap_testset([self.m, self.m2], self.rdms, method='cosine', fitter=None, N=100,
                          pattern_descriptor=None, rdm_descriptor=None)

    def test_bootstrap_testset_pattern(self):
        from rsatoolbox.inference import bootstrap_testset_pattern
        evaluations, n_cond = bootstrap_testset_pattern(
            [self.m, self.m2], self.rdms,
            method='cosine', fitter=None, N=100, pattern_descriptor=None)

    def test_bootstrap_testset_rdm(self):
        from rsatoolbox.inference import bootstrap_testset_rdm
        evaluations, n_rdms = bootstrap_testset_rdm(
            [self.m, self.m2], self.rdms,
            method='cosine', fitter=None, N=100, rdm_descriptor=None)


class TestSaveLoad(unittest.TestCase):

    def setUp(self) -> None:
        self.rng = np.random.default_rng(0)
        return super().setUp()

    def test_model_dict(self):
        from rsatoolbox.model import model_from_dict
        from rsatoolbox.model import ModelFixed
        m = ModelFixed('test1', self.rng.random(10))
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
        m = ModelSelect('test1', self.rng.random(10))
        model_dict = m.to_dict()
        model_loaded = model_from_dict(model_dict)
        assert m.name == model_loaded.name
        assert np.all(m.rdm_obj.dissimilarities
                      == model_loaded.rdm_obj.dissimilarities)

        from rsatoolbox.model import ModelWeighted
        m = ModelWeighted('test1', self.rng.random(10))
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


class TestResult(unittest.TestCase):
    def setUp(self):
        from rsatoolbox.inference import Result
        from rsatoolbox.model import ModelFixed
        self.rng = np.random.default_rng(0)
        m1 = ModelFixed('t1', np.arange(10))
        m2 = ModelFixed('t2', np.arange(2, 12))
        models = [m1, m2]
        method = 'corr'
        evaluations = np.array([[[0.9, 0.85, 0.87], [0.12, 0.11, 0.13]]])
        cv_method = 'fixed'
        noise_ceiling = [0.99, 0.99, 0.98]
        self.res = Result(models, evaluations, method, cv_method,
                          noise_ceiling, dof=2)

    def test_result_dict(self):
        from rsatoolbox.inference import Result
        from rsatoolbox.inference import result_from_dict
        from rsatoolbox.model import ModelFixed
        m1 = ModelFixed('test1', self.rng.random(10))
        m2 = ModelFixed('test2', self.rng.random(10))
        models = [m1, m2]
        evaluations = self.rng.random((100, 2))
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
            self.rng.random(10),
            pattern_descriptors={
                'test': ['test1', 'test1', 'test1', 'test3', 'test']})
        m1 = ModelFixed('test1', rdm)
        m2 = ModelFixed('test2', self.rng.random(10))
        models = [m1, m2]
        evaluations = self.rng.random((100, 2))
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

    def test_result_print(self):
        string = self.res.summary()
        self.assertEqual(
            string,
            'Results for running fixed evaluation for corr on 2 models:\n\n'
            + 'Model |   Eval ± SEM   | p (against 0) | p (against NC) |\n'
            + '---------------------------------------------------------\n'
            + 't1    |  0.873 ± 0.012 |      < 0.001  |         0.010  |\n'
            + 't2    |  0.120 ± 0.005 |      < 0.001  |       < 0.001  |\n\n'
            + 'p-values are based on uncorrected t-tests')


class TestsPairTests(unittest.TestCase):

    def setUp(self):
        self.rng = np.random.default_rng(0)
        self.evaluations = self.rng.random((100, 5, 10))

    def test_pair_tests(self):
        from rsatoolbox.util.inference_util import bootstrap_pair_tests
        ps = bootstrap_pair_tests(self.evaluations)
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

        rdms = RDMs(self.rng.random((11, 10)))  # 11 5x5 rdms
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

    def setUp(self) -> None:
        self.rng = np.random.default_rng(0)
        return super().setUp()

    def test_extract_var_1D(self):
        from rsatoolbox.util.inference_util import extract_variances
        variance = np.var(self.rng.standard_normal((10, 100)), 1)
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
        variance = np.cov(self.rng.standard_normal((10, 100)))
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
        variance = np.cov(self.rng.standard_normal((10, 100)))
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

    def test_zero_div(self):
        """This integration test makes sure things work when only 1 RDM is passed"""
        from rsatoolbox.rdm import RDMs
        from rsatoolbox.model import ModelFixed
        from rsatoolbox.inference import bootstrap_crossval
        model = ModelFixed('m', np.random.rand(190))
        data = RDMs(np.random.rand(190))
        result = bootstrap_crossval(model, data, boot_type="pattern", N=100)
        assert np.isfinite(result.model_var[0])
