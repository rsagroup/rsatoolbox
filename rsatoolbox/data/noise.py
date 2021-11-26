#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions for estimating the precision matrix based on the covariance of
either the residuals (temporal based precision matrix) or of the measurements
(instance based precision matrix)
"""

from collections.abc import Iterable
import numpy as np


def _check_demean(matrix):
    """
    checks that an input has 2 or 3 dimensions and subtracts the mean.
    returns a 2D matrix for covariance/precision computation and the
    degrees of freedom

    Args:
        matrix (np.ndarray):
            n_conditions x n_channels

    Returns:
        numpy.ndarray:
            demeaned matrix

    """
    assert isinstance(matrix, np.ndarray), "input must be ndarray"
    if matrix.ndim in [1, 2]:
        matrix = matrix - np.mean(matrix, axis=0, keepdims=True)
        dof = matrix.shape[0] - 1
    elif matrix.ndim == 3:
        matrix -= np.mean(matrix, axis=2, keepdims=True)
        matrix = matrix.transpose(0, 2, 1).reshape(
            matrix.shape[0] * matrix.shape[2], matrix.shape[1])
        dof = (matrix.shape[0] - 1) * matrix.shape[2]
    else:
        raise ValueError('Matrix for covariance estimation has wrong # of dimensions!')
    return matrix, dof


def _estimate_covariance(matrix, dof, method):
    """ calls the right covariance estimation function based on the ""method" argument

    Args:
        matrix (np.ndarray):
            n_conditions x n_channels

        dof (int):
            degrees of freedom

        method (string):
            which estimator to use

    Returns:
        numpy.ndarray, numpy.ndarray:
            cov_mat: n_channels x n_channels sample covariance matrix

    """
    matrix, dof_nat = _check_demean(matrix)
    if dof is None:
        dof = dof_nat
    # calculate sample covariance matrix s
    if method == 'shrinkage_eye':
        cov_mat = _covariance_eye(matrix, dof)
    elif method == 'shrinkage_diag':
        cov_mat = _covariance_diag(matrix, dof)
    elif method == 'diag':
        cov_mat = _variance(matrix)
    elif method == 'full':
        cov_mat = _covariance_full(matrix, dof)
    return cov_mat


def _variance(matrix, dof):
    """
    returns the vector of variances per measurement channel.
    The formula used here implies that the mean was already removed.

    Args:
        matrix (np.ndarray):
            n_conditions x n_channels

    Returns:
        numpy.ndarray:
            variance vector

    """
    return np.einsum('ij, ij-> j', matrix, matrix) / dof


def _covariance_full(matrix, dof):
    """
    computes the sample covariance matrix from a 2d-array.
    matrix should be demeaned before!

    Args:
        matrix (np.ndarray):
            n_conditions x n_channels

    Returns:
        numpy.ndarray, numpy.ndarray:
            s_mean: n_channels x n_channels sample covariance matrix

    """
    return np.einsum('ij, ik-> jk', matrix, matrix) / dof


def _covariance_eye(matrix, dof):
    """
    computes the sample covariance matrix from a 2d-array.
    matrix should be demeaned before!

    Computes an optimal shrinkage estimate of a sample covariance matrix
    as described by the following publication:

    Ledoit and Wolfe (2004): "A well-conditioned
    estimator for large-dimensional covariance matrices"

    Args:
        matrix (np.ndarray):
            n_conditions x n_channels

    Returns:
        numpy.ndarray, numpy.ndarray:
            s_mean: n_channels x n_channels sample covariance matrix

            xt_x:
            Einstein summation form of the matrix product
            of the 2d-array with itself

    """
    s_sum = np.zeros((matrix.shape[1], matrix.shape[1]))
    s2_sum = np.zeros((matrix.shape[1], matrix.shape[1]))
    for m_line in matrix:
        xt_x = np.outer(m_line, m_line)
        s_sum += xt_x
        s2_sum += xt_x ** 2
    s = s_sum / matrix.shape[0]
    b2 = np.sum(s2_sum / matrix.shape[0] - s * s) / matrix.shape[0]
    # calculate the scalar estimators to find the optimal shrinkage:
    # m, d^2, b^2 as in Ledoit & Wolfe paper
    m = np.sum(np.diag(s)) / s.shape[0]
    d2 = np.sum((s - m * np.eye(s.shape[0])) ** 2)
    b2 = min(d2, b2)
    # shrink covariance matrix
    s_shrink = b2 / d2 * m * np.eye(s.shape[0]) \
        + (d2-b2) / d2 * s
    # correction for degrees of freedom
    s_shrink = s_shrink * matrix.shape[0] / dof
    return s_shrink


def _covariance_diag(matrix, dof, mem_threshold=(10**9)/8):
    """
    computes the sample covariance matrix from a 2d-array.
    matrix should be demeaned before!

    Computes an optimal shrinkage estimate of a sample covariance matrix
    as described by the following publication:

    Schäfer, J., & Strimmer, K. (2005). "A Shrinkage Approach to Large-Scale
    Covariance Matrix Estimation and Implications for Functional Genomics.""

    Args:
        matrix (np.ndarray):
            n_conditions x n_channels

    Returns:
        numpy.ndarray, numpy.ndarray:
            s_mean: n_channels x n_channels sample covariance matrix

            xt_x:
            Einstein summation form of the matrix product
            of the 2d-array with itself

    """
    s_sum = np.zeros((matrix.shape[1], matrix.shape[1]))
    s2_sum = np.zeros((matrix.shape[1], matrix.shape[1]))
    for m_line in matrix:
        xt_x = np.outer(m_line, m_line)
        s_sum += xt_x
        s2_sum += xt_x ** 2
    s = s_sum / dof
    var = np.diag(s)
    std = np.sqrt(var)
    s_mean = s_sum / np.expand_dims(std, 0) / np.expand_dims(std, 1) / (matrix.shape[0] - 1)
    s2_mean = s2_sum / np.expand_dims(var, 0) / np.expand_dims(var, 1) / (matrix.shape[0] - 1)
    var_hat = matrix.shape[0] / dof ** 2 \
        * (s2_mean - s_mean ** 2)
    mask = ~np.eye(s.shape[0], dtype=np.bool)
    lamb = np.sum(var_hat[mask]) / np.sum(s_mean[mask] ** 2)
    lamb = max(min(lamb, 1), 0)
    scaling = np.eye(s.shape[0]) + (1-lamb) * mask
    s_shrink = s * scaling
    return s_shrink


def sample_covariance_3d(tensor):
    """
    computes the sample covariance matrix from a tensor by estimating the
    sample covariance for each slice along the third dimension and averaging
    the estimated covariance matrices.

    Args:
        tensor (numpy.ndarray):
            n_conditions x n_channels x n_measurements

    Returns:
         numpy.ndarray:
            s_mean: n_channels x n_channels expected sample covariance matrix

    """
    xt_x = np.einsum('ij, ik-> ijk', tensor, tensor)
    s = np.mean(xt_x, axis=0)
    return s, xt_x


def shrinkage_transform(s, xt_x, dof, target='eye'):
    """
    Computes an optimal shrinkage estimate of a sample covariance matrix
    as described by the following publications:

    Ledoit and Wolfe (2004): "A well-conditioned
    estimator for large-dimensional covariance matrices"

    or

    Schäfer, J., & Strimmer, K. (2005). "A Shrinkage Approach to Large-Scale
    Covariance Matrix Estimation and Implications for Functional Genomics.""

    Args:
        s(numpy.ndarray): covariance estimate
        xt_x(numpy.ndarray): sample wise xTx product to compute variances etc.
        dof(int or list of int): degrees of freedom for covariance estimation
            defaults to n_res - 1, should be corrected for the number
            of regressors in a GLM if applicable.
        target(str):
            switch to choose the shrinkage target
            'eye': multiple of the identity as a target. This is the simpler estimate
                described by Ledoit and Wolfe
            'diag': diagonal matrix as target. This is described by Schäfer & Strimmer.
                results in a matrix with correct diagonal and shrunk off-diagonal values

    Returns:
        numpy.ndarray: shrunk covariance estimate

    """
    if target == 'eye':
        # calculate the scalar estimators to find the optimal shrinkage:
        # m, d^2, b^2 as in Ledoit & Wolfe paper
        m = np.sum(np.diag(s)) / s.shape[0]
        d2 = np.sum((s - m * np.eye(s.shape[0])) ** 2)
        b2 = np.sum((xt_x - s) ** 2) / xt_x.shape[0] / xt_x.shape[0]
        b2 = min(d2, b2)
        # shrink covariance matrix
        s_shrink = b2 / d2 * m * np.eye(s.shape[0]) \
            + (d2-b2) / d2 * s
        # correction for degrees of freedom
        s_shrink = s_shrink * xt_x.shape[0] / dof
    elif target == 'diag':
        s = s * xt_x.shape[0] / dof
        var = np.diag(s)
        std = np.sqrt(var)
        xt_x_standard = xt_x / np.expand_dims(std, 0) / np.expand_dims(std, 1)
        xt_x_mean = np.mean(xt_x_standard, 0, keepdims=True)
        var_hat = xt_x.shape[0] / dof ** 2 / (xt_x.shape[0]-1) \
            * np.sum((xt_x_standard - xt_x_mean) ** 2, axis=0)
        mask = ~np.eye(s.shape[0], dtype=np.bool)
        lamb = np.sum(var_hat[mask]) / np.sum(xt_x_mean[0][mask] ** 2)
        lamb = max(min(lamb, 1), 0)
        scaling = np.eye(s.shape[0]) + (1-lamb) * mask
        s_shrink = s * scaling
    return s_shrink


def cov_from_residuals(residuals, dof=None, method='shrinkage_diag'):
    """
    Estimates a covariance matrix from measurements. Allows for shrinkage estimates.
    Use 'method' to choose which estimation method is used.

    Args:
        residuals(numpy.ndarray or list of these): n_residuals x n_channels
            matrix of residuals
        dof(int or list of int): degrees of freedom for covariance estimation
            defaults to n_res - 1, should be corrected for the number
            of regressors in a GLM if applicable.
        method(str): which estimate to use:
            'diag': provides a diagonal matrix, i.e. univariate noise normalizer
            'full': computes the sample covariance without shrinkage
            'shrinkage_eye': shrinks the data covariance towards a multiple of the identity.
            'shrinkage_diag': shrinks the covariance matrix towards the diagonal covariance matrix.

    Returns:
        numpy.ndarray (or list): sigma_p: covariance matrix over channels

    """
    if not isinstance(residuals, np.ndarray) or len(residuals.shape) > 2:
        cov_mat = []
        for i, residual in enumerate(residuals):
            if dof is None:
                cov_mat.append(cov_from_residuals(
                    residual, method=method))
            elif isinstance(dof, Iterable):
                cov_mat.append(cov_from_residuals(
                    residuals, method=method, dof=dof[i]))
            else:
                cov_mat.append(cov_from_residuals(
                    residual, method=method, dof=dof))
    else:
        cov_mat = _estimate_covariance(residuals, dof, method)
    return cov_mat


def prec_from_residuals(residuals, dof=None, method='shrinkage_diag'):
    """
    Estimates the covariance matrix from residuals and finds its multiplicative
    inverse (= the precision matrix)
    Use 'method' to choose which estimation method is used.

    Args:
        residuals(numpy.ndarray or list of these): n_residuals x n_channels
            matrix of residuals
        dof(int or list of int): degrees of freedom for covariance estimation
            defaults to n_res - 1, should be corrected for the number
            of regressors in a GLM if applicable.
        method(str): which estimate to use:
            'diag': provides a diagonal matrix, i.e. univariate noise normalizer
            'full': computes the sample covariance without shrinkage
            'shrinkage_eye': shrinks the data covariance towards a multiple of the identity.
            'shrinkage_diag': shrinks the covariance matrix towards the diagonal covariance matrix.

    Returns:
        numpy.ndarray (or list): sigma_p: precision matrix over channels

    """
    cov = cov_from_residuals(residuals=residuals, dof=dof, method=method)
    if not isinstance(cov, np.ndarray):
        prec = [None] * len(cov)
        for i, cov_i in enumerate(cov):
            prec[i] = np.linalg.inv(cov_i)
    elif len(cov.shape) > 2:
        prec = np.zeros(cov.shape)
        for i, cov_i in enumerate(cov):
            prec[i] = np.linalg.inv(cov_i)
    else:
        prec = np.linalg.inv(cov)
    return prec


def cov_from_measurements(dataset, obs_desc, dof=None, method='shrinkage_diag'):
    """
    Estimates a covariance matrix from measurements. Allows for shrinkage estimates.
    Use 'method' to choose which estimation method is used.

    Args:
        dataset(data.Dataset):
            rsatoolbox Dataset object
        dof(int or list of int): degrees of freedom for covariance estimation
            defaults to n_res - 1, should be corrected for the number
            of regressors in a GLM if applicable.
        method(str): which estimate to use:
            'diag': provides a diagonal matrix, i.e. univariate noise normalizer
            'full': computes the sample covariance without shrinkage
            'shrinkage_eye': shrinks the data covariance towards a multiple of the identity.
            'shrinkage_diag': shrinks the covariance matrix towards the diagonal covariance matrix.

    Returns:
        numpy.ndarray (or list): sigma_p: covariance matrix over channels

    """
    assert "Dataset" in str(type(dataset)), "Provided object is not a dataset"
    assert obs_desc in dataset.obs_descriptors.keys(), \
        "obs_desc not contained in the dataset's obs_descriptors"
    tensor, _ = dataset.get_measurements_tensor(obs_desc)
    # calculate sample covariance matrix s
    matrix, dof = _check_demean(tensor)
    if method == 'shrinkage_eye':
        s_mean, xt_x_mean = sample_covariance_3d(matrix)
        cov_mat = shrinkage_transform(s_mean, xt_x_mean, dof, target='eye')
    elif method == 'shrinkage_diag':
        s_mean, xt_x_mean = sample_covariance_3d(matrix)
        cov_mat = shrinkage_transform(s_mean, xt_x_mean, dof, target='diag')
    elif method == 'diag':
        cov_mat = _variance(matrix, dof)
    elif method == 'full':
        cov_mat = _covariance_full(matrix, dof)
    return cov_mat


def prec_from_measurements(dataset, obs_desc, dof=None, method='shrinkage_diag'):
    """
    Estimates the covariance matrix from measurements and finds its multiplicative
    inverse (= the precision matrix)
    Use 'method' to choose which estimation method is used.

    Args:
        residuals(numpy.ndarray or list of these): n_residuals x n_channels
            matrix of residuals
        dof(int or list of int): degrees of freedom for covariance estimation
            defaults to n_res - 1, should be corrected for the number
            of regressors in a GLM if applicable.
        method(str): which estimate to use:
            'diag': provides a diagonal matrix, i.e. univariate noise normalizer
            'full': computes the sample covariance without shrinkage
            'shrinkage_eye': shrinks the data covariance towards a multiple of the identity.
            'shrinkage_diag': shrinks the covariance matrix towards the diagonal covariance matrix.

    Returns:
        numpy.ndarray (or list): sigma_p: precision matrix over channels

    """
    cov = cov_from_measurements(dataset, obs_desc, dof=dof, method=method)
    if not isinstance(cov, np.ndarray):
        prec = [None] * len(cov)
        for i, cov_i in enumerate(cov):
            prec[i] = np.linalg.inv(cov_i)
    elif len(cov.shape) > 2:
        prec = np.zeros(cov.shape)
        for i, cov_i in enumerate(cov):
            prec[i] = np.linalg.inv(cov_i)
    else:
        prec = np.linalg.inv(cov)
    return prec
