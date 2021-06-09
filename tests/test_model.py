#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for the model subpackage
"""

import unittest
import rsatoolbox.model as model
import numpy as np
from numpy.testing import assert_allclose


class TestModel(unittest.TestCase):
    """ Tests for the Model superclass
    """

    def test_creation(self):
        _ = model.Model('Test Model')


class TestModelFixed(unittest.TestCase):
    """ Tests for the fixed model class
    """

    def test_creation(self):
        rdm = np.array(np.ones(6))
        m = model.ModelFixed('Test Model', rdm)
        m.fit([])
        pred = m.predict()
        assert np.all(pred == rdm)

    def test_creation_rdm(self):
        from rsatoolbox.rdm import RDMs
        rdm = np.array(np.ones(6))
        rdm_obj = RDMs(np.array([rdm]))
        m = model.ModelFixed('Test Model', rdm_obj)
        m.fit(rdm_obj)
        pred = m.predict()
        assert np.all(pred == rdm)
        pred_obj = m.predict_rdm()
        assert isinstance(pred_obj, RDMs)


class TestModelSelect(unittest.TestCase):
    """ Tests for the fixed model class
    """

    def test_creation(self):
        rdm = np.random.rand(2, 6)
        m = model.ModelSelect('Test Model', rdm)
        pred = m.predict()
        assert np.all(pred == rdm[0])

    def test_creation_rdm(self):
        from rsatoolbox.rdm import RDMs
        rdm = np.random.rand(2, 6)
        pattern_descriptors = {'test': ['a', 'b', 'c', 'd']}
        rdm_obj = RDMs(rdm, dissimilarity_measure='euclid',
                       pattern_descriptors=pattern_descriptors)
        m = model.ModelSelect('Test Model', rdm_obj)
        pred = m.predict()
        assert np.all(pred == rdm[0])
        pred_obj = m.predict_rdm()
        assert isinstance(pred_obj, RDMs)
        assert pred_obj.n_rdm == 1
        assert pred_obj.pattern_descriptors == pattern_descriptors

    def test_fit(self):
        from rsatoolbox.rdm import RDMs
        rdm = np.random.rand(2, 6)
        pattern_descriptors = {'test': ['a', 'b', 'c', 'd']}
        rdm_descriptors = {'ind': np.array([1, 2])}
        rdm_obj = RDMs(rdm, dissimilarity_measure='euclid',
                       pattern_descriptors=pattern_descriptors,
                       rdm_descriptors=rdm_descriptors)
        m = model.ModelSelect('Test Model', rdm_obj)
        train = rdm_obj.subset('ind', 2)
        theta = m.fit(train)
        assert theta == 1


class TestModelWeighted(unittest.TestCase):
    """ Tests for the fixed model class
    """

    def test_creation(self):
        rdm = np.random.rand(4, 15)
        m = model.ModelWeighted('Test Model', rdm)
        pred = m.predict([1, 0, 0, 0])
        assert np.all(pred == rdm[0])

    def test_creation_rdm(self):
        from rsatoolbox.rdm import RDMs
        rdm = np.random.rand(2, 6)
        pattern_descriptors = {'test': ['a', 'b', 'c', 'd']}
        rdm_obj = RDMs(rdm, dissimilarity_measure='euclid',
                       pattern_descriptors=pattern_descriptors)
        m = model.ModelWeighted('Test Model', rdm_obj)
        pred = m.predict(np.array([1, 0]))
        assert np.all(pred == rdm[0])
        pred_obj = m.predict_rdm()
        assert isinstance(pred_obj, RDMs)
        assert pred_obj.n_rdm == 1
        assert pred_obj.pattern_descriptors == pattern_descriptors

    def test_fit(self):
        from rsatoolbox.rdm import RDMs
        rdm = np.random.rand(2, 6)
        pattern_descriptors = {'test': ['a', 'b', 'c', 'd']}
        rdm_descriptors = {'ind': np.array([1, 2])}
        rdm_obj = RDMs(rdm, dissimilarity_measure='euclid',
                       pattern_descriptors=pattern_descriptors,
                       rdm_descriptors=rdm_descriptors)
        m = model.ModelWeighted('Test Model', rdm_obj)
        train = rdm_obj.subset('ind', 2)
        theta = m.fit(train)


class TestModelInterpolate(unittest.TestCase):
    """ Tests for the fixed model class
    """

    def test_creation(self):
        rdm = np.random.rand(4, 15)
        m = model.ModelInterpolate('Test Model', rdm)
        pred = m.predict([1, 0, 0, 0])
        assert np.all(pred == rdm[0])

    def test_creation_rdm(self):
        from rsatoolbox.rdm import RDMs
        rdm = np.random.rand(2, 6)
        pattern_descriptors = {'test': ['a', 'b', 'c', 'd']}
        rdm_obj = RDMs(rdm, dissimilarity_measure='euclid',
                       pattern_descriptors=pattern_descriptors)
        m = model.ModelInterpolate('Test Model', rdm_obj)
        pred = m.predict(np.array([1, 0]))
        assert np.all(pred == rdm[0])
        pred_obj = m.predict_rdm()
        assert isinstance(pred_obj, RDMs)
        assert pred_obj.n_rdm == 1
        assert pred_obj.pattern_descriptors == pattern_descriptors

    def test_fit(self):
        from rsatoolbox.rdm import RDMs
        rdm = np.random.rand(5, 15)
        pattern_descriptors = {'test': ['a', 'b', 'c', 'd', 'e', 'f']}
        rdm_descriptors = {'ind': np.array([1, 2, 3, 1, 2])}
        rdm_obj = RDMs(rdm, dissimilarity_measure='euclid',
                       pattern_descriptors=pattern_descriptors,
                       rdm_descriptors=rdm_descriptors)
        m = model.ModelInterpolate('Test Model', rdm_obj)
        train = rdm_obj.subset('ind', 2)
        theta = m.fit(train)
        _ = m.predict(theta)


class TestConsistency(unittest.TestCase):
    """ Tests which compare different model types and fitting methods,
    which should be equivalent
    """

    def setUp(self):
        from rsatoolbox.data import Dataset
        from rsatoolbox.rdm import calc_rdm
        from rsatoolbox.rdm import concat
        rdms = []
        for _ in range(5):
            data = np.random.rand(6, 20)
            data_s = Dataset(data)
            rdms.append(calc_rdm(data_s))
        self.rdms = concat(rdms)

    def test_two_rdms(self):
        from rsatoolbox.model import ModelInterpolate, ModelWeighted
        from rsatoolbox.model.fitter import fit_regress, fit_optimize_positive
        from rsatoolbox.rdm import concat, compare
        model_rdms = concat([self.rdms[0], self.rdms[1]])
        model_weighted = ModelWeighted(
            'm_weighted',
            model_rdms)
        model_interpolate = ModelInterpolate(
            'm_interpolate',
            model_rdms)
        for i_method in ['cosine', 'corr', 'cosine_cov', 'corr_cov']:
            theta_m_i = model_interpolate.fit(self.rdms, method=i_method)
            theta_m_w = model_weighted.fit(self.rdms, method=i_method)
            theta_m_w_pos = fit_optimize_positive(
                model_weighted, self.rdms, method=i_method)
            theta_m_w_linear = fit_regress(
                model_weighted, self.rdms, method=i_method)
            eval_m_i = np.mean(compare(model_weighted.predict_rdm(
                theta_m_i), self.rdms, method=i_method))
            eval_m_w = np.mean(compare(model_weighted.predict_rdm(
                theta_m_w), self.rdms, method=i_method))
            eval_m_w_pos = np.mean(compare(model_weighted.predict_rdm(
                theta_m_w_pos), self.rdms, method=i_method))
            eval_m_w_linear = np.mean(compare(model_weighted.predict_rdm(
                theta_m_w_linear), self.rdms, method=i_method))
            self.assertAlmostEqual(
                eval_m_i, eval_m_w_pos,
                places=4, msg='weighted fit differs from interpolation fit!'
                + '\nfor %s' % i_method)
            self.assertAlmostEqual(
                eval_m_w, eval_m_w_linear,
                places=4, msg='regression fit differs from optimization fit!'
                + '\nfor %s' % i_method)

    def test_two_rdms_nan(self):
        from rsatoolbox.model import ModelInterpolate, ModelWeighted
        from rsatoolbox.model.fitter import fit_regress, fit_optimize_positive
        from rsatoolbox.rdm import concat, compare
        rdms = self.rdms.subsample_pattern('index', [0, 1, 1, 3, 4, 5])
        model_rdms = concat([rdms[0], rdms[1]])
        model_weighted = ModelWeighted(
            'm_weighted',
            model_rdms)
        model_interpolate = ModelInterpolate(
            'm_interpolate',
            model_rdms)
        for i_method in ['cosine', 'corr', 'cosine_cov', 'corr_cov']:
            theta_m_i = model_interpolate.fit(rdms, method=i_method)
            theta_m_w = model_weighted.fit(rdms, method=i_method)
            theta_m_w_pos = fit_optimize_positive(
                model_weighted, rdms, method=i_method)
            theta_m_w_linear = fit_regress(
                model_weighted, rdms, method=i_method)
            eval_m_i = np.mean(compare(model_weighted.predict_rdm(
                theta_m_i), rdms, method=i_method))
            eval_m_w = np.mean(compare(model_weighted.predict_rdm(
                theta_m_w), rdms, method=i_method))
            eval_m_w_pos = np.mean(compare(model_weighted.predict_rdm(
                theta_m_w_pos), rdms, method=i_method))
            eval_m_w_linear = np.mean(compare(model_weighted.predict_rdm(
                theta_m_w_linear), rdms, method=i_method))
            self.assertAlmostEqual(
                eval_m_i, eval_m_w_pos,
                places=4, msg='weighted fit differs from interpolation fit!'
                + '\nfor %s' % i_method)
            self.assertAlmostEqual(
                eval_m_w, eval_m_w_linear,
                places=4, msg='regression fit differs from optimization fit!'
                + '\nfor %s' % i_method)


class TestNNLS(unittest.TestCase):
    """ Tests that the non-negative least squares give results consistent
    with other solutions where they apply
    """

    def test_nnls_scipy(self):
        from scipy.optimize import nnls
        from rsatoolbox.model.fitter import _nn_least_squares
        A = np.random.rand(10, 3)
        b = A @ np.array([1, -0.1, -0.1])
        x_scipy, loss_scipy = nnls(A, b)
        x_rsatoolbox, loss_rsatoolbox = _nn_least_squares(A, b)
        assert_allclose(
            x_scipy, x_rsatoolbox,
            err_msg='non-negative-least squares different from scipy')
        self.assertAlmostEqual(
            loss_scipy, np.sqrt(loss_rsatoolbox),
            places=5, msg='non-negative-least squares different from scipy')

    def test_nnls_eye(self):
        from rsatoolbox.model.fitter import _nn_least_squares
        A = np.random.rand(10, 3)
        b = A @ np.array([1, -0.1, -0.1])
        x_rsatoolbox, loss_rsatoolbox = _nn_least_squares(A, b)
        x_rsatoolbox_v, loss_rsatoolbox_v = _nn_least_squares(A, b, V=np.eye(10))
        assert_allclose(
            x_rsatoolbox, x_rsatoolbox_v,
            err_msg='non-negative-least squares changes with V=np.eye')
        self.assertAlmostEqual(
            loss_rsatoolbox_v, loss_rsatoolbox,
            places=5, msg='nnls loss changes with np.eye')

    def test_nnls_eye_ridge(self):
        from rsatoolbox.model.fitter import _nn_least_squares
        A = np.random.rand(10, 3)
        b = A @ np.array([1, -0.1, -0.1])
        x_rsatoolbox, loss_rsatoolbox = _nn_least_squares(A, b, ridge_weight=1)
        x_rsatoolbox_v, loss_rsatoolbox_v = _nn_least_squares(
            A, b, ridge_weight=1, V=np.eye(10))
        assert_allclose(
            x_rsatoolbox, x_rsatoolbox_v,
            err_msg='non-negative-least squares changes with V=np.eye')
        self.assertAlmostEqual(
            loss_rsatoolbox_v, loss_rsatoolbox,
            places=5, msg='nnls loss changes with np.eye')
