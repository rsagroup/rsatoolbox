#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Collection of helper methods for rdm module

@author: baihan
"""

import numpy as np
from scipy.spatial.distance import squareform


def batch_to_vectors(rdms):
    """converts a *stack* of RDMs in vector or matrix form into vector form

    Args:
        rdms: stack of RDMs

    Returns:
        tuple: **vectors** (np.ndarray): 2D, vector form of the stack of RDMs

        **n_rdm** (int): number of rdms

        **n_cond** (int): number of conditions
    """
    if rdms.ndim == 2:
        vectors = rdms
        n_rdm = rdms.shape[0]
        n_cond = _get_n_from_length(rdms.shape[1])
    elif rdms.ndim == 3:
        matrices = rdms
        n_rdm = rdms.shape[0]
        n_cond = rdms.shape[1]
        vectors = np.ndarray((n_rdm, int(n_cond * (n_cond - 1) / 2)))
        for idx in np.arange(n_rdm):
            vectors[idx, :] = squareform(matrices[idx, :, :], checks=False)
    elif rdms.ndim == 1:
        vectors = np.array([rdms])
        n_rdm = 1
        n_cond = _get_n_from_length(vectors.shape[1])
    return vectors, n_rdm, n_cond


def batch_to_matrices(rdms):
    """converts a *stack* of RDMs in vector or matrix form into matrix form

    Args:
        **rdms**: stack of RDMs

    Returns:
        tuple: **v** (np.ndarray): 3D, matrix form of the stack of RDMs

        **n_rdm** (int): number of rdms

        **n_cond** (int): number of conditions
    """
    if rdms.ndim == 2:
        vectors = rdms
        n_rdm = vectors.shape[0]
        n_cond = _get_n_from_length(rdms.shape[1])
        matrices = np.ndarray((n_rdm, n_cond, n_cond))
        for idx in np.arange(n_rdm):
            matrices[idx, :, :] = squareform(vectors[idx, :])
    elif rdms.ndim == 3:
        matrices = rdms
        n_rdm = rdms.shape[0]
        n_cond = rdms.shape[1]
    return matrices, n_rdm, n_cond


def _get_n_from_reduced_vectors(x):
    """
    calculates the size of the RDM from the vector representation

    Args:
        **shape**(np.ndarray): vector representation

    Returns:
        int: n: number of conditions

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
