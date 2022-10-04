#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 23:58:28 2020

@author: heiko
"""

import numpy as np
import scipy.sparse
from scipy.stats import rankdata
from rsatoolbox.rdm import RDMs
from rsatoolbox.util.matrix import get_v


def pool_rdm(rdms, method='cosine', sigma_k=None):
    """pools multiple RDMs into the one with maximal performance under a given
    evaluation metric
    rdm_descriptors of the generated rdms are empty

    Args:
        rdms (pyrsa.rdm.RDMs):
            RDMs to be pooled
        method : String, optional
            Which comparison method to optimize for. The default is 'cosine'.

    Returns:
        pyrsa.rdm.RDMs: the pooled RDM, i.e. a RDM with maximal performance
            under the chosen method

    """
    rdm_vec = rdms.get_vectors()
    if method == 'euclid':
        rdm_vec = _nan_mean(rdm_vec)
    elif method == 'cosine':
        rdm_vec = rdm_vec / np.sqrt(np.nanmean(rdm_vec ** 2, axis=1,
                                               keepdims=True))
        rdm_vec = _nan_mean(rdm_vec)
    elif method == 'corr':
        rdm_vec = rdm_vec - np.nanmean(rdm_vec, axis=1, keepdims=True)
        rdm_vec = rdm_vec / np.nanstd(rdm_vec, axis=1, keepdims=True)
        rdm_vec = _nan_mean(rdm_vec)
        rdm_vec = rdm_vec - np.nanmin(rdm_vec) + 0.01
    elif method == 'cosine_cov':
        v = get_v(rdms.n_cond, sigma_k=sigma_k)
        ok_idx = np.all(np.isfinite(rdm_vec), axis=0)
        v = v[ok_idx][:, ok_idx]
        rdm_vec_nonan = rdm_vec[:, ok_idx]
        v_inv_x = np.array([scipy.sparse.linalg.cg(v, rdm_vec_nonan[i],
                                                   atol=10 ** -9)[0]
                            for i in range(rdms.n_rdm)])
        rdm_norms = np.einsum('ij, ij->i', rdm_vec_nonan, v_inv_x).reshape(
            [rdms.n_rdm, 1])
        rdm_vec = rdm_vec / np.sqrt(rdm_norms)
        rdm_vec = _nan_mean(rdm_vec)
    elif method == 'corr_cov':
        rdm_vec = rdm_vec - np.nanmean(rdm_vec, axis=1, keepdims=True)
        v = get_v(rdms.n_cond, sigma_k=sigma_k)
        ok_idx = np.all(np.isfinite(rdm_vec), axis=0)
        v = v[ok_idx][:, ok_idx]
        rdm_vec_nonan = rdm_vec[:, ok_idx]
        v_inv_x = np.array([scipy.sparse.linalg.cg(v, rdm_vec_nonan[i],
                                                   atol=10 ** -9)[0]
                            for i in range(rdms.n_rdm)])
        rdm_norms = np.einsum('ij, ij->i', rdm_vec_nonan, v_inv_x).reshape(
            [rdms.n_rdm, 1])
        rdm_vec = rdm_vec / np.sqrt(rdm_norms)
        rdm_vec = _nan_mean(rdm_vec)
        rdm_vec = rdm_vec - np.nanmin(rdm_vec) + 0.01
    elif method in ('spearman', 'rho-a'):
        rdm_vec = np.array([_nan_rank_data(v) for v in rdm_vec])
        rdm_vec = _nan_mean(rdm_vec)
    elif method == 'rho-a':
        rdm_vec = np.array([_nan_rank_data(v) for v in rdm_vec])
        rdm_vec = _nan_mean(rdm_vec)
    elif method in ('kendall', 'tau-b', 'tau-a'):
        Warning('Noise ceiling for tau based on averaged ranks!')
        rdm_vec = np.array([_nan_rank_data(v) for v in rdm_vec])
        rdm_vec = _nan_mean(rdm_vec)
    else:
        raise ValueError('Unknown RDM comparison method requested!')
    return RDMs(rdm_vec,
                dissimilarity_measure=rdms.dissimilarity_measure,
                descriptors=rdms.descriptors,
                rdm_descriptors=None,
                pattern_descriptors=rdms.pattern_descriptors)


def _nan_mean(rdm_vector):
    """ takes the average over a rdm_vector with nans for masked entries
    without a warning

    Args:
        rdm_vector(numpy.ndarray): set of rdm_vectors to be averaged

    Returns:
        rdm_mean(numpy.ndarray): the mean rdm

    """
    nan_idx = ~np.isnan(rdm_vector[0])
    mean_values = np.mean(rdm_vector[:, nan_idx], axis=0)
    rdm_mean = np.empty((1, rdm_vector.shape[1])) * np.nan
    rdm_mean[:, nan_idx] = mean_values
    return rdm_mean


def _nan_rank_data(rdm_vector):
    """ rank_data for vectors with nan entries

    Args:
        rdm_vector(numpy.ndarray): the vector to be rank_transformed

    Returns:
        ranks(numpy.ndarray): the ranks with nans where the original vector
            had nans

    """
    ranks_no_nan = rankdata(rdm_vector[~np.isnan(rdm_vector)])
    ranks = np.ones_like(rdm_vector) * np.nan
    ranks[~np.isnan(rdm_vector)] = ranks_no_nan
    return ranks
