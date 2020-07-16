#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Definition of visualization methods
@author: baihan
"""

import numpy as np
from sklearn.manifold import MDS

sd = np.random.RandomState(seed=1)
mds_emb = MDS(n_components=2, random_state=sd, dissimilarity="precomputed")


def rdm_dimension_reduction(rdms, func):
    """ dimension reduction of RDMs class

    Args:
        rdms (RDMs class): an RDMs class object
        func (function): an sklearn transform function

    Returns:
        dr (numpy.ndarray): a dimension-reduced
        embedding of size (n_rdm x n_cond x n_emb)

    """
    rdmm = rdms.get_matrices()
    drs = np.ndarray((rdms.n_rdm, rdms.n_cond, 2))
    for i in np.arange(rdms.n_rdm):
        drs[i, :, :] = func.fit_transform(rdmm[i, :, :])
    return drs


def mds(rdms):
    """ multi-dimensional scaling of RDMs class

    Args:
        rdms (RDMs class): an RDMs class object

    Returns:
        (numpy.ndarray): an MDS embedding

    """
    return rdm_dimension_reduction(rdms, mds_emb)
