#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parameter fitting methods for models
"""

import numpy as np
import scipy.optimize as opt
import scipy.sparse
from rsatoolbox.rdm import compare
from rsatoolbox.util.matrix import get_v
from rsatoolbox.util.pooling import pool_rdm
from rsatoolbox.util.rdm_utils import _parse_nan_vectors


class Fitter:
    """Object to specify a fitting function and parameters

    Effectively this gives the user a convenient way to specify fitting
    functions with different settings than the defaults.

    Create this object with the fitting function to use and additional
    keyword arguments for settings you wish to change. The resulting
    object then behaves as the fitting function itself with the
    keyword arguments set to the different values provided at object
    creation.

    Example:
        generate Fitter-object for ridge regression:

        ::

            fit = pyrsa.model.Fitter(pyrsa.model.fit_regress, ridge_weight=1)

        the resulting object 'fit' now does the same as
        pyrsa.model.fit_regress when run with the additional argument
        ``ridge_weight=1``, i.e. the following two lines now yield equal results:

        ::

            fit(model, data)
            pyrsa.model.fit_regress(model, data, ridge_weight=1)

    For a general introduction to flexible models see demo_flex.
    """

    def __init__(self, fit_fun, **kwargs):
        self.fit_fun = fit_fun
        self.kwargs = kwargs

    def __call__(self, model, data, *args, **more_args):
        return self.fit_fun(model, data, *args, **more_args, **self.kwargs)


def fit_mock(model, data, method='cosine', pattern_idx=None,
             pattern_descriptor=None, sigma_k=None):
    """ formally acceptable fitting method which always returns a vector of
    zeros

    Args:
        model(rsatoolbox.model.Model): model to be fit
        data(rsatoolbox.rdm.RDMs): Data to fit to
        method(String): Evaluation method
        pattern_idx(numpy.ndarray): Which patterns are sampled
        pattern_descriptor(String): Which descriptor is used
        sigma_k(matrix): pattern-covariance matrix
            used only for whitened distances (ending in _cov)
            to compute the covariance matrix for rdms

    Returns:
        theta(numpy.ndarray): parameter vector

    """
    return np.zeros(model.n_param)


def fit_select(model, data, method='cosine', pattern_idx=None,
               pattern_descriptor=None, sigma_k=None):
    """ fits selection models by evaluating each rdm and selcting the one
    with best performance. Works only for ModelSelect

    Args:
        model(rsatoolbox.model.Model): model to be fit
        data(rsatoolbox.rdm.RDMs): Data to fit to
        method(String): Evaluation method
        pattern_idx(numpy.ndarray): Which patterns are sampled
        pattern_descriptor(String): Which descriptor is used
        sigma_k(matrix): pattern-covariance matrix
            used only for whitened distances (ending in _cov)
            to compute the covariance matrix for rdms

    Returns:
        theta(int): parameter vector

    """
    evaluations = np.zeros(model.n_rdm)
    for i_rdm in range(model.n_rdm):
        pred = model.predict_rdm(i_rdm)
        if not (pattern_idx is None or pattern_descriptor is None):
            pred = pred.subsample_pattern(pattern_descriptor, pattern_idx)
        evaluations[i_rdm] = np.mean(
            compare(pred, data, method=method, sigma_k=sigma_k))
    theta = np.argmax(evaluations)
    return theta


def fit_optimize(model, data, method='cosine', pattern_idx=None,
                 pattern_descriptor=None, sigma_k=None, ridge_weight=0,
                 normalize=True):
    """
    fitting theta using optimization
    currently allowed for ModelWeighted only

    Args:
        model(Model): the model to be fit
        data(rsatoolbox.rdm.RDMs): data to be fit
        method(String, optional): evaluation metric The default is 'cosine'.
        pattern_idx(numpy.ndarray, optional)
            sampled patterns The default is None.
        pattern_descriptor (String, optional)
            descriptor used for fitting. The default is None.
        sigma_k(matrix): pattern-covariance matrix
            used only for whitened distances (ending in _cov)
            to compute the covariance matrix for rdms
        normalize(bool): whether to normalize the theta vector
            default = True
            If true, theta is normalized to norm 1.
            This is sensible for many models where the norm
            of theta does not vary the loss.

    Returns:
        numpy.ndarray: theta, parameter vector for the model

    """
    def _loss_opt(theta):
        return _loss(theta, model, data, method=method,
                     pattern_idx=pattern_idx,
                     pattern_descriptor=pattern_descriptor,
                     sigma_k=sigma_k, ridge_weight=ridge_weight)
    thetas = []
    losses = []
    for _ in range(2 * model.n_param):
        theta0 = np.random.rand(model.n_param)
        theta = opt.minimize(
            _loss_opt,
            theta0,
            method='BFGS',
            tol=0.000001
        )
        thetas.append(theta.x)
        losses.append(theta.fun)
    id = np.argmin(losses)
    theta = thetas[id]
    if not normalize:
        return theta.flatten()
    norm = np.sum(theta ** 2)
    if norm == 0:
        return theta.flatten()
    return theta.flatten() / np.sqrt(norm)


def fit_optimize_positive(
        model, data, method='cosine', pattern_idx=None,
        pattern_descriptor=None, sigma_k=None, ridge_weight=0,
        normalize=True):
    """
    fitting theta using optimization enforcing positive weights
    currently allowed for ModelWeighted only

    Args:
        model(Model): the model to be fit
        data(pyrsa.rdm.RDMs): data to be fit
        method(String, optional): evaluation metric The default is 'cosine'.
        pattern_idx(numpy.ndarray, optional)
            sampled patterns The default is None.
        pattern_descriptor (String, optional)
            descriptor used for fitting. The default is None.
        sigma_k(matrix): pattern-covariance matrix
            used only for whitened distances (ending in _cov)
            to compute the covariance matrix for rdms
        normalize(bool): whether to normalize the theta vector
            default = True
            If true, theta is normalized to norm 1.
            This is sensible for many models where the norm
            of theta does not vary the loss.

    Returns:
        numpy.ndarray: theta, parameter vector for the model

    """
    def _loss_opt(theta):
        return _loss(theta ** 2, model, data, method=method,
                     pattern_idx=pattern_idx,
                     pattern_descriptor=pattern_descriptor,
                     sigma_k=sigma_k, ridge_weight=ridge_weight)
    theta0 = np.zeros(model.n_param)
    thetas = [theta0]
    losses = [_loss_opt(theta0)]
    theta0 = np.random.rand(model.n_param)
    theta = opt.minimize(
        fun=_loss_opt,
        x0=theta0,
        method='BFGS',
        tol=0.000001
    )
    thetas.append(theta.x)
    losses.append(theta.fun)
    for i in range(model.n_param):
        theta0 = np.ones(model.n_param) * 0.001
        theta0[i] = 1
        theta = opt.minimize(
            fun=_loss_opt,
            x0=theta0,
            method='BFGS',
            tol=0.000001
        )
        thetas.append(theta.x)
        losses.append(theta.fun)
    id = np.argmin(losses)
    theta = thetas[id] ** 2
    if not normalize:
        return theta.flatten()
    norm = np.sum(theta ** 2)
    if norm == 0:
        return theta.flatten()
    return theta.flatten() / np.sqrt(norm)


def fit_interpolate(model, data, method='cosine', pattern_idx=None,
                    pattern_descriptor=None, sigma_k=None):
    """
    fitting theta using bisection optimization
    allowed for ModelInterpolate only

    Args:
        model(Model): the model to be fit
        data(rsatoolbox.rdm.RDMs): data to be fit
        method(String, optional): evaluation metric The default is 'cosine'.
        pattern_idx(numpy.ndarray, optional)
            sampled patterns The default is None.
        pattern_descriptor (String, optional)
            descriptor used for fitting. The default is None.
        sigma_k(matrix): pattern-covariance matrix
            used only for whitened distances (ending in _cov)
            to compute the covariance matrix for rdms

    Returns:
        numpy.ndarray: theta, parameter vector for the model

    """
    results = []
    for i_pair in range(model.n_rdm - 1):
        def loss_opt(w):
            theta = np.zeros(model.n_param)
            theta[i_pair] = w
            theta[i_pair + 1] = 1 - w
            return _loss(theta, model, data, method=method,
                         pattern_idx=pattern_idx,
                         pattern_descriptor=pattern_descriptor,
                         sigma_k=sigma_k)
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
                pattern_descriptor=None, ridge_weight=0, sigma_k=None,
                normalize=True):
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
        normalize(bool): whether to normalize the theta vector
            default = True
            If true, theta is normalized to norm 1.
            This is sensible for many models where the norm
            of theta does not vary the loss.

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
    vectors, y, nan_idx = _parse_nan_vectors(vectors, y)
    # Normalizations
    if method == 'cosine':
        v = None
    elif method == 'corr':
        vectors = vectors - np.mean(vectors, 1, keepdims=True)
        v = None
    elif method == 'cosine_cov':
        v = get_v(pred.n_cond, sigma_k)
        v = v[nan_idx[0]][:, nan_idx[0]]
    elif method == 'corr_cov':
        vectors = vectors - np.mean(vectors, 1, keepdims=True)
        y = y - np.mean(y)
        v = get_v(pred.n_cond, sigma_k)
        v = v[nan_idx[0]][:, nan_idx[0]]
    else:
        raise ValueError('method argument invalid')
    if v is None:
        X = vectors @ vectors.T + ridge_weight * np.eye(vectors.shape[0])
        y = vectors @ y.T
    else:
        v_inv_x = np.array([scipy.sparse.linalg.cg(v, vectors[i],
                                                   atol=10 ** -9)[0]
                            for i in range(vectors.shape[0])])
        y = v_inv_x @ y.T
        X = vectors @ v_inv_x.T + ridge_weight * np.eye(vectors.shape[0])
    theta = np.linalg.solve(X, y)
    if not normalize:
        return theta.flatten()
    norm = np.sum(theta ** 2)
    if norm == 0:
        return theta.flatten()
    return theta.flatten() / np.sqrt(np.sum(theta ** 2))


def fit_regress_nn(model, data, method='cosine', pattern_idx=None,
                   pattern_descriptor=None, ridge_weight=0, sigma_k=None,
                   normalize=True):
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
        normalize(bool): whether to normalize the theta vector
            default = True
            If true, theta is normalized to norm 1.
            This is sensible for many models where the norm
            of theta does not vary the loss.

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
    vectors, y, non_nan_mask = _parse_nan_vectors(vectors, y)
    # Normalizations
    if method == 'cosine':
        v = None
    elif method == 'corr':
        vectors = vectors - np.mean(vectors, 1, keepdims=True)
        v = None
    elif method == 'cosine_cov':
        v = get_v(pred.n_cond, sigma_k)
        v = v[non_nan_mask[0]][:, non_nan_mask[0]]
    elif method == 'corr_cov':
        vectors = vectors - np.mean(vectors, 1, keepdims=True)
        y = y - np.mean(y)
        v = get_v(pred.n_cond, sigma_k)
        v = v[non_nan_mask[0]][:, non_nan_mask[0]]
    else:
        raise ValueError('method argument invalid')
    theta, _ = _nn_least_squares(vectors.T, y[0], ridge_weight=ridge_weight, V=v)
    if not normalize:
        return theta.flatten()
    norm = np.sum(theta ** 2)
    if norm == 0:
        return theta.flatten()
    return theta.flatten() / np.sqrt(np.sum(theta ** 2))


def _loss(theta, model, data, method='cosine', sigma_k=None,
          pattern_descriptor=None, pattern_idx=None,
          ridge_weight=0):
    """Method for calculating a loss for a model and parameter combination

    Args:
        theta(numpy.ndarray): evaluated parameter value
        model(Model): the model to be fit
        data(rsatoolbox.rdm.RDMs): data to be fit
        method(String, optional): evaluation metric The default is 'cosine'.
        pattern_idx(numpy.ndarray, optional)
            sampled patterns The default is None.
        pattern_descriptor (String, optional)
            descriptor used for fitting. The default is None.
        sigma_k(matrix): pattern-covariance matrix
            used only for whitened distances (ending in _cov)
            to compute the covariance matrix for rdms
        ridge_weight(float): weight for a ridge regularisation

    Returns:

        numpy.ndarray: loss

    """
    pred = model.predict_rdm(theta)
    if not (pattern_idx is None or pattern_descriptor is None):
        pred = pred.subsample_pattern(pattern_descriptor, pattern_idx)
    return -np.mean(compare(pred, data, method=method, sigma_k=sigma_k)) \
        + np.sum(theta * theta) * ridge_weight


def _nn_least_squares(A, y, ridge_weight=0, V=None):
    """ non-negative least squares
    essentially scipy.optimize.nnls extended to accept a ridge_regression
    regularisation and/or a covariance matrix V.

    The algorithm is discribed in detail here:
    Bro, R., & Jong, S. D. (1997). A fast non-negativity-constrained
    least squares algorithm. Journal of Chemometrics, 11, 9.


    This is an active set algorithm which is somewhat optimized by
    precomputing A^T V^-1 A and A^T V y such that during the optimization
    only matricies of rank r need to be inverted.

    This is tested against the scipy solution for ridge_weight=0 and V=None.
    For other V the validation comes from fitting the same models using
    general optimization.
    """
    assert A.shape[0] == y.shape[0]
    assert y.ndim == 1
    x = np.zeros(A.shape[1])
    p = np.zeros(A.shape[1], bool)
    if V is None:
        w = A.T @ y
        ATA = A.T @ A + ridge_weight * np.eye(A.shape[1])
    else:
        V_A = np.array([scipy.sparse.linalg.cg(V, A[:, i],
                                               atol=10 ** -9)[0]
                        for i in range(A.shape[1])])
        y_V_A = V_A @ y
        w = y_V_A
        ATA = A.T @ V_A.T + ridge_weight * np.eye(A.shape[1])
    while np.max(w) > 100 * np.finfo(float).eps:
        p[np.argmax(w)] = True
        if V is None:
            s_p = np.linalg.solve(ATA[p][:, p], A[:, p].T @ y)
        else:
            s_p = np.linalg.solve(ATA[p][:, p], y_V_A[p])
        while np.any(s_p < 0):
            alphas = x[p] / (x[p] - s_p)
            alphas[s_p > 0] = 1
            i_alpha = np.argmin(alphas)
            alpha = alphas[i_alpha]
            x[p] = x[p] + alpha * (s_p - x[p])
            i_alpha = np.where(p)[0][i_alpha]
            x[i_alpha] = 0
            p[i_alpha] = False
            if V is None:
                s_p = np.linalg.solve(ATA[p][:, p], A[:, p].T @ y)
            else:
                s_p = np.linalg.solve(ATA[p][:, p], y_V_A[p])
        x[p] = s_p
        if V is None:
            w = A.T @ y - ATA @ x
        else:
            w = y_V_A - ATA @ x
    if V is None:
        loss = np.sum((y - A @ x) ** 2)
    else:
        loss = (y - A @ x).T @ V @ (y - A @ x)
    return x, loss
