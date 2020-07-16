#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 12:42:06 2020

@author: heiko
"""

from collections.abc import Iterable
import numpy as np


def cov_from_residuals(residuals, dof=None):
    """
    computes an optimal shrinkage estimate of the precision matrix from
    the residuals as described by Ledoit and Wolfe (2004): "A well-conditioned
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
    if not isinstance(residuals, np.ndarray) or len(residuals.shape) > 2:
        s_shrink = []
        for i in range(len(residuals)):
            if dof is None:
                s_shrink.append(cov_from_residuals(residuals[i]))
            elif isinstance(dof, Iterable):
                s_shrink.append(cov_from_residuals(residuals[i], dof[i]))
            else:
                s_shrink.append(cov_from_residuals(residuals[i], dof))
    else:
        if dof is None:
            dof = residuals.shape[0] - 1
        # calculate sample covariance matrix s
        residuals = residuals - np.mean(residuals, axis=0, keepdims=True)
        xt_x = np.einsum('ij, ik-> ijk', residuals, residuals)
        s = np.sum(xt_x, axis=0) / xt_x.shape[0]
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


def prec_from_residuals(residuals, dof=None):
    """
    computes an optimal shrinkage estimate of the precision matrix from
    the residuals as described by Ledoit and Wolfe (2004): "A well-conditioned
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
    cov = cov_from_residuals(residuals=residuals, dof=dof)
    if not isinstance(cov, np.ndarray) or len(cov.shape) > 2:
        for i in range(len(cov)):
            cov[i] = np.linalg.inv(cov[i])
    else:
        cov = np.linalg.inv(cov)
    return cov
