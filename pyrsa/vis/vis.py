#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Definition of visualization methods
@author: baihan
"""

import pyrsa as rsa
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from sklearn.decomposition import PCA

sd = np.random.RandomState(seed=1)
mds = MDS(n_components=2, random_state=sd, dissimilarity="precomputed")

def dimension_reduction(rdms, func):
    """ dimension reduction of RDMs class

    Args:
        rdms (RDMs class): an RDMs class object
        func (function): an sklearn transform function

    Returns:
        dr (numpy.ndarray): a dimension-reduced 
        embedding

    """
    rdmm = rdms.get_matrices()
    dr = func.fit_transform(rdmm)
    return dr

def mds(rdms):
    """ multi-dimensional scaling of RDMs class

    Args:
        rdms (RDMs class): an RDMs class object

    Returns:
        (numpy.ndarray): an MDS embedding

    """
    return dimension_reduction(rdms,mds)

