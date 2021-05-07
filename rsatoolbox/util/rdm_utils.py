#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Collection of helper methods for rdm module

@author: baihan
"""

from typing import Tuple
import numpy as np
from numpy import sqrt, nan, inf, ndarray
from scipy.spatial.distance import squareform


def batch_to_vectors(x):
    """converts a *stack* of RDMs in vector or matrix form into vector form

    Args:
        x: stack of RDMs

    Returns:
        tuple: **v** (np.ndarray): 2D, vector form of the stack of RDMs

        **n_rdm** (int): number of rdms

        **n_cond** (int): number of conditions
    """
    if x.ndim == 2:
        v = x
        n_rdm = x.shape[0]
        n_cond = _get_n_from_reduced_vectors(x)
    elif x.ndim == 3:
        m = x
        n_rdm = x.shape[0]
        n_cond = x.shape[1]
        v = np.ndarray((n_rdm, int(n_cond * (n_cond - 1) / 2)))
        for idx in np.arange(n_rdm):
            v[idx, :] = squareform(m[idx, :, :], checks=False)
    elif x.ndim == 1:
        v = np.array([x])
        n_rdm = 1
        n_cond = _get_n_from_reduced_vectors(v)
    return v, n_rdm, n_cond


def batch_to_matrices(x):
    """converts a *stack* of RDMs in vector or matrix form into matrix form

    Args:
        **x**: stack of RDMs

    Returns:
        tuple: **v** (np.ndarray): 3D, matrix form of the stack of RDMs

        **n_rdm** (int): number of rdms

        **n_cond** (int): number of conditions
    """
    if x.ndim == 2:
        v = x
        n_rdm = x.shape[0]
        n_cond = _get_n_from_reduced_vectors(x)
        m = np.ndarray((n_rdm, n_cond, n_cond))
        for idx in np.arange(n_rdm):
            m[idx, :, :] = squareform(v[idx, :])
    elif x.ndim == 3:
        m = x
        n_rdm = x.shape[0]
        n_cond = x.shape[1]
    return m, n_rdm, n_cond


def _get_n_from_reduced_vectors(x):
    """
    calculates the size of the RDM from the vector representation

    Args:
        **x**(np.ndarray): stack of RDM vectors (2D)

    Returns:
        int: n: size of the RDM

    """
    return max(int(np.ceil(np.sqrt(x.shape[1] * 2))), 1)


def _get_n_from_length(n):
    """
    calculates the size of the RDM from the vector length

    Args:
        **x**(np.ndarray): stack of RDM vectors (2D)

    Returns:
        int: n: size of the RDM

    """
    return int(np.ceil(np.sqrt(n * 2)))


def add_pattern_index(rdms, pattern_descriptor):
    """
    adds index if pattern_descriptor is None

    Args:
        **rdms** (rsatoolbox.rdm.RDMs): rdms object to be parsed

    Returns:
        pattern_descriptor
        pattern_select

    """
    pattern_select = rdms.pattern_descriptors[pattern_descriptor]
    pattern_select = np.unique(pattern_select)
    return pattern_descriptor, pattern_select


def _parse_input_rdms(rdm1, rdm2):
    """Gets the vector representation of input RDMs, raises an error if
    the two RDMs objects have different dimensions

    Args:
        rdm1 (pyrsa.rdm.RDMs):
            first set of RDMs
        rdm2 (pyrsa.rdm.RDMs):
            second set of RDMs

    """
    if not isinstance(rdm1, np.ndarray):
        vector1 = rdm1.get_vectors()
    else:
        if len(rdm1.shape) == 1:
            vector1 = rdm1.reshape(1, -1)
        else:
            vector1 = rdm1
    if not isinstance(rdm2, np.ndarray):
        vector2 = rdm2.get_vectors()
    else:
        if len(rdm2.shape) == 1:
            vector2 = rdm2.reshape(1, -1)
        else:
            vector2 = rdm2
    if not vector1.shape[1] == vector2.shape[1]:
        raise ValueError('rdm1 and rdm2 must be RDMs of equal shape')
    nan_idx = ~np.isnan(vector1)
    vector1_no_nan = vector1[nan_idx].reshape(vector1.shape[0], -1)
    vector2_no_nan = vector2[~np.isnan(vector2)].reshape(vector2.shape[0], -1)
    if not vector1_no_nan.shape[1] == vector2_no_nan.shape[1]:
        raise ValueError('rdm1 and rdm2 have different nan positions')
    return vector1_no_nan, vector2_no_nan, nan_idx


def _extract_triu_(X):
    """ extracts the upper triangular vector as a masked view

    Args:
        X (numpy.ndarray): 2D symmetric matrix

    Returns:
        vector version of X

    """
    mask = np.triu(np.ones_like(X, dtype=bool), k=1)
    return X[mask]


def _mean(vectors:ndarray, weights:ndarray=None) -> ndarray:
    """Weighted mean of RDM vectors, ignores nans

    See :meth:`rsatoolbox.rdm.rdms.RDMs.mean`

    Args:
        vectors (ndarray): dissimilarity vectors of shape (nrdms, nconds)
        weights (ndarray, optional): Same shape as vectors.

    Returns:
        ndarray: Average vector of shape (nconds,)
    """
    if weights is None:
        weights = np.ones(vectors.shape)
        weights[np.isnan(vectors)] = np.nan
    weighted_sum = np.nansum(vectors * weights, axis=0)
    return weighted_sum / np.nansum(weights, axis=0)


def _ss(vectors:ndarray) -> ndarray:
    """Sum of squares on the last dimension

    Args:
        vectors (ndarray): 1- or 2-dimensional data

    Returns:
        ndarray: the sum of squares, with an extra empty dimension
    """
    summed_squares = np.nansum(vectors ** 2, axis=vectors.ndim-1)
    return np.expand_dims(summed_squares, axis=vectors.ndim-1)


def _scale(vectors:ndarray) -> ndarray:
    """Divide by the root sum of squares

    Args:
        vectors (ndarray): 1- or 2-dimensional data

    Returns:
        ndarray: input scaled
    """
    return vectors / sqrt(_ss(vectors))


def _rescale(dissim:ndarray, method:str) -> Tuple[ndarray, ndarray]:
    """Rescale RDM vectors

    See :meth:`rsatoolbox.rdm.combine.rescale`

    Args:
        dissim (ndarray): dissimilarity vectors, shape = (rdms, conds)
        method (str): one of 'evidence', 'setsize' or 'simple'.

    Returns:
        (ndarray, ndarray): Tuple of the aligned dissimilarity vectors
            and the weights used
    """
    n_rdms, n_conds = dissim.shape
    if method == 'evidence':
        weights = (dissim ** 2).clip(0.2 ** 2)
    elif method == 'setsize':
        setsize = np.isfinite(dissim).sum(axis=1)
        weights = np.tile(1 / setsize, [n_conds, 1]).T
    else:
        weights = np.ones(dissim.shape)
    weights[np.isnan(dissim)] = np.nan

    current_estimate = _scale(_mean(dissim))
    prev_estimate = np.full([n_conds,], -inf)
    while _ss(current_estimate - prev_estimate) > 1e-8:
        prev_estimate = current_estimate.copy()
        tiled_estimate = np.tile(current_estimate, [n_rdms, 1])
        tiled_estimate[np.isnan(dissim)] = nan
        aligned = _scale(dissim) * sqrt(_ss(tiled_estimate))
        current_estimate = _scale(_mean(aligned, weights))

    return aligned, weights
