#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Definition of visualization methods

+ rdm_dimension_reduction:  dimension reduction of RDMs class 
+ mds:  multi-dimensional scaling of RDMs class
+ weighted_mds:  weighted multi-dimensional scaling of RDMs class

@author: baihan
"""

import numpy as np
from sklearn.manifold import MDS
from pyrsa.util import Weighted_MDS

sd = np.random.RandomState(seed=1)

def rdm_dimension_reduction(rdms, func, dim=2, weight=None):
    """ dimension reduction of RDMs class

    Args:
        rdms (RDMs class): an RDMs class object
        func (function): an sklearn transform function

    Returns:
        dr (numpy.ndarray): a dimension-reduced
        embedding of size (n_rdm x n_cond x n_emb)

    """
    rdmm = rdms.get_matrices()
    drs = np.ndarray((rdms.n_rdm, rdms.n_cond, dim))
    for i in np.arange(rdms.n_rdm):
        drs[i, :, :] = func.fit_transform(rdmm[i, :, :])
    return drs


def mds(rdms, dim=2):
    """ multi-dimensional scaling of RDMs class

    Args:
        rdms (RDMs class): an RDMs class object

    Returns:
        (numpy.ndarray): an MDS embedding

    """
    mds_emb = MDS(n_components=dim, 
                  random_state=sd, 
                  dissimilarity="precomputed")
    return rdm_dimension_reduction(rdms, mds_emb)


def weighted_mds(rdms, dim=2, weight=None):
    """ multi-dimensional scaling of RDMs class

    Args:
        rdms (RDMs class): an RDMs class object

    Returns:
        (numpy.ndarray): an MDS embedding

    """
    wmds_emb = Weighted_MDS(n_components=dim, 
                            random_state=sd, 
                            dissimilarity="precomputed")
    return rdm_dimension_reduction(rdms, wmds_emb)