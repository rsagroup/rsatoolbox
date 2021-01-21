#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions for estimating the precision matrix based on the covariance of
either the residuals (temporal based precision matrix) or of the measurements
(instance based precision matrix)
"""

from collections.abc import Iterable
import numpy as np


def sample_covariance(matrix):
    """
    computes the sample covariance matrix from a 2d-array

    Args:
        matrix (np.ndarray):
            n_conditions x n_channels

    Returns:
        s_mean (np.ndarray):
            n_channels x n_channels sample covariance matrix
        xt_x (np.ndarray):
            Einstein summation form of the matrix product
            from the 2d-array with itself

    """
    assert isinstance(matrix, np.ndarray), "input must be ndarray"
    assert len(matrix.shape) == 2, "input must have 2 dimensions"
    # calculate sample covariance matrix s
    matrix = matrix - np.mean(matrix, axis=0, keepdims=True)
    xt_x = np.einsum('ij, ik-> ijk', matrix, matrix)
    s = np.mean(xt_x, axis=0)
    return s, xt_x


def sample_covariance_3d(tensor):
    """
    computes the sample covariance matrix from a tensor by estimating the
    sample covariance for each slice along the third dimension and averaging
    the estimated covariance matrices.

    Args:
        tensor (np.ndarray):
            n_conditions x n_channels x n_measurements

    Returns:
        s_mean (np.ndarray):
            n_channels x n_channels expected sample covariance matrix

    """
    assert isinstance(tensor, np.ndarray), "input must be ndarray"
    assert len(tensor.shape) == 3, "input must have 3 dimensions"

    # calculate sample covariance matrix s for each slice of the tensor
    einsum_superset = []
    cov_superset = []
    for slice_num in range(tensor.shape[2]):
        array_slice = tensor[:, :, slice_num]
        s, xt_x = sample_covariance(array_slice)
        einsum_superset.append(xt_x)
        cov_superset.append(s)

    # get expected value of the covariance matrix estimates
    einsum_tensor = np.stack(einsum_superset, axis=0)
    xt_x_mean = np.mean(einsum_tensor, axis=0)
    s_mean = np.mean(xt_x_mean, axis=0)
    return s_mean, xt_x_mean


def shrinkage_transform(s, xt_x, dof):
    """
    Computes an optimal shrinkage estimate of a sample covariance matrix
    as described by Ledoit and Wolfe (2004): "A well-conditioned
    estimator for large-dimensional covariance matrices"

    Args:
        residuals(numpy.ndarray or list of these): n_residuals x n_channels
            matrix of residuals
        dof(int or list of int): degrees of freedom for covariance estimation
            defaults to n_res - 1, should be corrected for the number
            of regressors in a GLM if applicable.

    Returns:
        numpy.ndarray (or list): sigma_p: covariance matrix over channels

    """
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
    return s_shrink


def cov_from_residuals(residuals, dof=None):
    """
    Computes a covariance matrix for residuals and applies a shrinkage
    transform

    Args:
        residuals(numpy.ndarray or list of these): n_residuals x n_channels
            matrix of residuals
        dof(int or list of int): degrees of freedom for covariance estimation
            defaults to n_res - 1, should be corrected for the number
            of regressors in a GLM if applicable.

    Returns:
        numpy.ndarray (or list): sigma_p: covariance matrix over channels

    """
    if not isinstance(residuals, np.ndarray) or len(residuals.shape) > 2:
        s_shrink = []
        for i in range(len(residuals)):
            if dof is None:
                s_shrink.append(cov_from_residuals(residuals[i]))
            elif isinstance(dof, Iterable):
                s_shrink.append(
                    cov_from_residuals(residuals[i], dof[i]))
            else:
                s_shrink.append(
                    cov_from_residuals(residuals[i], dof))
    else:
        if dof is None:
            dof = residuals.shape[0] - 1
        # calculate sample covariance matrix s
        s, xt_x = sample_covariance(residuals)
        # apply shrinkage transform
        s_shrink = shrinkage_transform(s, xt_x, dof)
    return s_shrink


def prec_from_residuals(residuals, dof=None):
    """
    Computes a covariance matrix for residuals, applies a shrinkage
    transform to it and finds its multiplicative inverse
    (= the precision matrix)

    Args:
        residuals(numpy.ndarray or list of these): n_residuals x n_channels
            matrix of residuals
        dof(int or list of int): degrees of freedom for covariance estimation
            defaults to n_res - 1, should be corrected for the number
            of regressors in a GLM if applicable.

    Returns:
        numpy.ndarray (or list): sigma_p: precision matrix over channels

    """
    cov = cov_from_residuals(residuals=residuals, dof=dof)
    if not isinstance(cov, np.ndarray) or len(cov.shape) > 2:
        prec = [None] * len(cov)
        for i in range(len(cov)):
            prec[i] = np.linalg.inv(cov[i])
    else:
        prec = np.linalg.inv(cov)
    return prec


def cov_from_measurements(dataset, obs_desc, dof=None):
    """
    Computes a covariance matrix for measurements and applies a shrinkage
    transform

    Args:
        dataset(data.Dataset):
            PyRSA Dataset object
        dof(int or list of int): degrees of freedom for covariance estimation
            defaults to n_res - 1, should be corrected for the number
            of regressors in a GLM if applicable.

    Returns:
        numpy.ndarray (or list): sigma_p: covariance matrix over channels

    """
    assert "Dataset" in str(type(dataset)), "Provided object is not a dataset"
    assert obs_desc in dataset.obs_descriptors.keys(), \
        "obs_desc not contained in the dataset's obs_descriptors"
    tensor, _ = dataset.get_measurements_tensor(obs_desc)
    if dof is None:
        dof = tensor.shape[0] * tensor.shape[2] - 1
    # calculate sample covariance matrix s
    s_mean, xt_x_mean = sample_covariance_3d(tensor)
    # apply shrinkage transform
    s_shrink = shrinkage_transform(s_mean, xt_x_mean, dof)
    return s_shrink


def prec_from_measurements(dataset, obs_desc, dof=None):
    """
    Computes a covariance matrix for measurements, applies a shrinkage
    transform to it and finds its inverse, i.e. the precision matrix

    Args:
        residuals(numpy.ndarray or list of these): n_residuals x n_channels
            matrix of residuals
        dof(int or list of int): degrees of freedom for covariance estimation
            defaults to n_res - 1, should be corrected for the number
            of regressors in a GLM if applicable.

    Returns:
        numpy.ndarray (or list): sigma_p: precision matrix over channels

    """
    cov = cov_from_measurements(dataset, obs_desc, dof=dof)
    prec = np.zeros(cov.shape)
    if not isinstance(cov, np.ndarray) or len(cov.shape) > 2:
        for i in range(len(cov)):
            prec[i] = np.linalg.inv(cov[i])
    else:
        prec = np.linalg.inv(cov)
    return prec
