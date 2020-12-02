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
        n_cond = _get_n_from_reduced_vectors(rdms.shape[1])
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
        n_cond = _get_n_from_reduced_vectors(vectors.shape[1])
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
        n_cond = _get_n_from_reduced_vectors(rdms.shape[1])
        matrices = np.ndarray((n_rdm, n_cond, n_cond))
        for idx in np.arange(n_rdm):
            matrices[idx, :, :] = squareform(vectors[idx, :])
    elif rdms.ndim == 3:
        matrices = rdms
        n_rdm = rdms.shape[0]
        n_cond = rdms.shape[1]
    return matrices, n_rdm, n_cond


def _get_n_from_reduced_vectors(shape):
    """
    calculates the size of the RDM from the vector representation

    Args:
        **shape**(int): length of the vector representation

    Returns:
        int: n: number of conditions

    """
    return int(np.ceil(np.sqrt(shape * 2)))


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
