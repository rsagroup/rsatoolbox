#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parameter fitting methods for models
"""

import numpy as np
import scipy.optimize as opt
import scipy.sparse
from pyrsa.rdm import compare
from pyrsa.rdm.compare import _get_v
from pyrsa.util.inference_util import pool_rdm


def fit_mock(model, data, method='cosine', pattern_idx=None,
             pattern_descriptor=None):
    """ formally acceptable fitting method which always returns a vector of
    zeros

    Args:
        model(pyrsa.model.Model): model to be fit
        data(pyrsa.rdm.RDMs): Data to fit to
        method(String): Evaluation method
        pattern_idx(numpy.ndarray): Which patterns are sampled
        pattern_descriptor(String): Which descriptor is used

    Returns:
        theta(numpy.ndarray): parameter vector

    """
    return np.zeros(model.n_param)


def fit_select(model, data, method='cosine', pattern_idx=None,
               pattern_descriptor=None):
    """ fits selection models by evaluating each rdm and selcting the one
    with best performance. Works only for ModelSelect

    Args:
        model(pyrsa.model.Model): model to be fit
        data(pyrsa.rdm.RDMs): Data to fit to
        method(String): Evaluation method
        pattern_idx(numpy.ndarray): Which patterns are sampled
        pattern_descriptor(String): Which descriptor is used

    Returns:
        theta(int): parameter vector

    """
    evaluations = np.zeros(model.n_rdm)
    for i_rdm in range(model.n_rdm):
        pred = model.predict_rdm(i_rdm)
        if not (pattern_idx is None or pattern_descriptor is None):
            pred = pred.subsample_pattern(pattern_descriptor, pattern_idx)
        evaluations[i_rdm] = np.mean(compare(pred, data, method=method))
    theta = np.argmax(evaluations)
    return theta


def fit_optimize(model, data, method='cosine', pattern_idx=None,
                 pattern_descriptor=None):
    """
    fitting theta using optimization
    currently allowed for ModelWeighted only

    Args:
        model(Model): the model to be fit
        data(pyrsa.rdm.RDMs): data to be fit
        method(String, optional): evaluation metric The default is 'cosine'.
        pattern_idx(numpy.ndarray, optional)
            sampled patterns The default is None.
        pattern_descriptor (String, optional)
            descriptor used for fitting. The default is None.

    Returns:
        numpy.ndarray: theta, parameter vector for the model

    """
    def _loss_opt(theta):
        return _loss(theta, model, data, method=method,
                     pattern_idx=pattern_idx,
                     pattern_descriptor=pattern_descriptor)
    theta0 = np.random.rand(model.n_param)
    theta = opt.minimize(_loss_opt, theta0)
    return theta.x


def fit_interpolate(model, data, method='cosine', pattern_idx=None,
                    pattern_descriptor=None):
    """
    fitting theta using bisection optimization
    allowed for ModelInterpolate only

    Args:
        model(Model): the model to be fit
        data(pyrsa.rdm.RDMs): data to be fit
        method(String, optional): evaluation metric The default is 'cosine'.
        pattern_idx(numpy.ndarray, optional)
            sampled patterns The default is None.
        pattern_descriptor (String, optional)
            descriptor used for fitting. The default is None.

    Returns:
        numpy.ndarray: theta, parameter vector for the model

    """
    results = []
    for i_pair in range(model.n_rdm-1):
        def loss_opt(w):
            theta = np.zeros(model.n_param)
            theta[i_pair] = w
            theta[i_pair + 1] = 1 - w
            return _loss(theta, model, data, method=method,
                         pattern_idx=pattern_idx,
                         pattern_descriptor=pattern_descriptor)
        results.append(
            opt.minimize_scalar(loss_opt, np.array([.5]),
                                method='bounded', bounds=(0, 1)))
    losses = [r.fun for r in results]
    i_pair = np.argmin(losses)
    result = results[i_pair]
    theta = np.zeros(model.n_rdm)
    theta[i_pair] = result.x
    theta[i_pair + 1] = 1 - result.x
    return theta


def fit_regress(model, data, method='cosine', pattern_idx=None,
                pattern_descriptor=None, ridge_weight=0, sigma_k=None):
    """
    fitting theta using linear algebra solutions to the OLS problem
    allowed for ModelWeighted only
    This method first normalizes the data and model RDMs appropriately
    for the measure to be optimized. For 'cosine' similarity this is a
    normalization of the data-RDMs to vector length 1. For correlation
    the mean is removed from both model and data rdms additionally.
    Then the parameters are estimated using ordinary least squares.

    Args:
        model(Model): the model to be fit
        data(pyrsa.rdm.RDMs): data to be fit
        method(String, optional): evaluation metric The default is 'cosine'.
        pattern_idx(numpy.ndarray, optional)
            sampled patterns The default is None.
        pattern_descriptor (String, optional)
            descriptor used for fitting. The default is None.
        ridge_weight (float, default=0)
            weight for the ridge-regularisation of the regression
            weight is in comparison to the final regression problem on
            the appropriately normalized regressors
        sigma_k(matrix): pattern-covariance matrix
            used only for whitened distances (ending in _cov)
            to compute the covariance matrix for rdms

    Returns:
        numpy.ndarray: theta, parameter vector for the model

    """
    if not (pattern_idx is None or pattern_descriptor is None):
        pred = model.rdm_obj.subsample_pattern(pattern_descriptor, pattern_idx)
    else:
        pred = model.rdm_obj
    vectors = pred.get_vectors()
    data_mean = pool_rdm(data, method=method)
    y = data_mean.get_vectors()
    # Normalizations
    if method == 'cosine':
        v = None
    elif method == 'corr':
        vectors = vectors - np.mean(vectors, 1, keepdims=True)
        v = None
    elif method == 'corr_cov':
        vectors = vectors - np.mean(vectors, 1, keepdims=True)
        v = _get_v(pred.n_cond, sigma_k)
    else:
        raise ValueError('method argument invalid')
    if v is None:
        X = vectors @ vectors.T + ridge_weight * np.eye(vectors.shape[0])
    else:
        v_inv_x = np.array([scipy.sparse.linalg.cg(v, vectors[i])[0]
                            for i in range(vectors.shape[0])])
        X = vectors @ v_inv_x.T + ridge_weight * np.eye(vectors.shape[0])
    theta = np.linalg.solve(X, vectors @ y.T)
    return theta


def _loss(theta, model, data, method='cosine', cov=None,
          pattern_descriptor=None, pattern_idx=None):
    """Method for calculating a loss for a model and parameter combination

    Args:
        theta(numpy.ndarray): evaluated parameter value
        model(Model): the model to be fit
        data(pyrsa.rdm.RDMs): data to be fit
        method(String, optional): evaluation metric The default is 'cosine'.
        pattern_idx(numpy.ndarray, optional)
            sampled patterns The default is None.
        pattern_descriptor (String, optional)
            descriptor used for fitting. The default is None.
        cov(numpy.ndarray, optional):
            Covariance matrix for likelihood based evaluation.
            It is ignored otherwise. The default is None.

    Returns:

        numpy.ndarray: loss

    """
    pred = model.predict_rdm(theta)
    if not (pattern_idx is None or pattern_descriptor is None):
        pred = pred.subsample_pattern(pattern_descriptor, pattern_idx)
    return -np.mean(compare(pred, data, method=method))
