#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Collection of helper methods for rdm module
    batch_to_vectors:  batch squareform() to vectors
    batch_to_matrices: batch squareform() to matrices
    get_unique_unsorted: return a unique unsorted list
@author: baihan
"""

import numpy as np
import pyrsa as rsa
from scipy.spatial.distance import squareform


def batch_to_vectors(x):
    if x.ndim == 2:
        v = x
        n_rdm = x.shape[0]
        n_cond = get_n_from_reduced_vectors(x)
    elif x.ndim == 3:
        m = x
        n_rdm = x.shape[0]
        n_cond = x.shape[1]
        v = np.ndarray((n_rdm, int(n_cond * (n_cond - 1) / 2)))
        for idx in np.arange(n_rdm):
            v[idx, :] = squareform(m[idx, :, :])
    return v, n_rdm, n_cond


def batch_to_matrices(x):
    if x.ndim == 2:
        v = x
        n_rdm = x.shape[0]
        n_cond = get_n_from_reduced_vectors(x)
        m = np.ndarray((n_rdm, n_cond, n_cond))
        for idx in np.arange(n_rdm):
            m[idx, :, :] = squareform(v[idx, :])
    elif x.ndim == 3:
        m = x
        n_rdm = x.shape[0]
        n_cond = x.shape[1]
    return m, n_rdm, n_cond


def get_n_from_reduced_vectors(x):
    return int(np.ceil(np.sqrt(x.shape[1] * 2)))
