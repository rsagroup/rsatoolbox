#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dimensional reduction transformations and visualizations (with potential
integrations of additional plotting options)

@author: baihan
"""
from __future__ import annotations
from typing import TYPE_CHECKING, Optional
import numpy as np
from sklearn.manifold import MDS
from rsatoolbox.util.vis_utils import weight_to_matrices, Weighted_MDS
from rsatoolbox.vis.scatter_plot import show_scatter
if TYPE_CHECKING:
    from rsatoolbox.rdm import RDMs
    from numpy.typing import NDArray
    from matplotlib.figure import Figure
sd = np.random.RandomState(seed=1)


def rdm_dimension_reduction(rdms, func, dim=2, weight=None):
    """ dimension reduction base function of RDMs class

    Args:
        **rdms** (rsatoolbox.rdm.RDMs): an RDMs class object
        **func** (function): an sklearn transform function

    Returns:
        (np.ndarray): a dimension-reduced embedding of
            size (n_rdm x n_cond x n_dim)
    """
    rdmm = rdms.get_matrices()
    ws = weight_to_matrices(weight) if weight is not None else None
    drs = np.ndarray((rdms.n_rdm, rdms.n_cond, dim))
    for i in np.arange(rdms.n_rdm):
        if weight is not None:
            drs[i, :, :] = func.fit_transform(rdmm[i, :, :],
                                              weight=ws[i, :, :])
        else:
            drs[i, :, :] = func.fit_transform(rdmm[i, :, :])
    return drs


def mds(rdms, dim=2, weight=None):
    """ multi-dimensional scaling of RDMs class

    Args:
        **rdms** (RDMs class): an RDMs class object

        **dim** (int): the dimension of MDS embedding

        **weight** (np.ndarray): an importance matrix for distances

    Returns:
        (np.ndarray): an MDS embedding of shape (rdms, conds, dims)
    """
    emb = MDS if weight is None else Weighted_MDS
    mds_emb = emb(n_components=dim,
                  random_state=sd,
                  dissimilarity="precomputed")
    return rdm_dimension_reduction(rdms, mds_emb, dim, weight)


def show_mds(rdms: RDMs, pattern_descriptor: Optional[str]) -> Figure:
    coords = mds(rdms)
    return show_scatter(
        rdms,
        coords,
        pattern_descriptor=pattern_descriptor
    )
