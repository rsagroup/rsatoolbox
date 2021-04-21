#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Collection of helper methods for rdm module

@author: baihan
"""

import numpy as np
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
        **rdms** (pyrsa.rdm.RDMs): rdms object to be parsed

    Returns:
        pattern_descriptor
        pattern_select

    """
    pattern_select = rdms.pattern_descriptors[pattern_descriptor]
    pattern_select = np.unique(pattern_select)
    return pattern_descriptor, pattern_select


def _extract_triu_(X):
    """ extracts the upper triangular vector as a masked view

    Args:
        X (numpy.ndarray): 2D symmetric matrix

    Returns:
        vector version of X

    """
    mask = np.triu(np.ones_like(X, dtype=bool), k=1)
    return X[mask]
