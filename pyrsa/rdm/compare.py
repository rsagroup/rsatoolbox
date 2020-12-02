#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comparison methods for comparing two RDMs objects
"""
import numpy as np
import scipy.stats
from scipy.stats._stats import _kendall_dis
from pyrsa.util.matrix import pairwise_contrast_sparse
from pyrsa.util.rdm_utils import _get_n_from_reduced_vectors
from pyrsa.util.matrix import row_col_indicator_g


def compare(rdm1, rdm2, method='cosine', sigma_k=None):
    """calculates the similarity between two RDMs objects using a chosen method

    Args:
        rdm1 (pyrsa.rdm.RDMs):
            first set of RDMs
        rdm2 (pyrsa.rdm.RDMs):
            second set of RDMs
        method (string):
            which method to use, options are:
            'cosine' = cosine similarity
            'spearman' = spearman rank correlation
            'corr' = pearson correlation
            'kendall' = kendall-tau b
            'tau-a' = kendall-tau a
            'rho-a' = spearman correlation without tie correction
            'corr_cov' = pearson correlation after whitening
            'cosine_cov' = unbiased distance correlation
                which is equivalent to the cosine dinstance after whitening
        sigma_k (numpy.ndarray):
            covariance matrix of the pattern estimates
            Used only for corr_cov and cosine_cov

    Returns:
        numpy.ndarray: dist:
            pariwise similarities between the RDMs from the RDMs objects

    """
    if method == 'cosine':
        sim = compare_cosine(rdm1, rdm2)
    elif method == 'spearman':
        sim = compare_spearman(rdm1, rdm2)
    elif method == 'corr':
        sim = compare_correlation(rdm1, rdm2)
    elif method == 'kendall' or method == 'tau-b':
        sim = compare_kendall_tau(rdm1, rdm2)
    elif method == 'tau-a':
        sim = compare_kendall_tau_a(rdm1, rdm2)
    elif method == 'rho-a':
        sim = compare_rho_a(rdm1, rdm2)
    elif method == 'corr_cov':
        sim = compare_correlation_cov_weighted(rdm1, rdm2, sigma_k=sigma_k)
    elif method == 'cosine_cov':
        sim = compare_cosine_cov_weighted(rdm1, rdm2, sigma_k=sigma_k)
    else:
        raise ValueError('Unknown RDM comparison method requested!')
    return sim


def compare_cosine(rdm1, rdm2):
    """calculates the cosine similarities between two RDMs objects

    Args:
        rdm1 (pyrsa.rdm.RDMs):
            first set of RDMs
        rdm2 (pyrsa.rdm.RDMs):
            second set of RDMs
    Returns:
        numpy.ndarray: dist
            cosine similarity between the two RDMs

    """
    vector1, vector2 = _parse_input_rdms(rdm1, rdm2)
    sim = _cosine(vector1, vector2)
    return sim


def compare_correlation(rdm1, rdm2):
    """calculates the correlations between two RDMs objects

    Args:
        rdm1 (pyrsa.rdm.RDMs):
            first set of RDMs
        rdm2 (pyrsa.rdm.RDMs):
            second set of RDMs
    Returns:
        numpy.ndarray: dist:
            correlations between the two RDMs

    """
    vector1, vector2 = _parse_input_rdms(rdm1, rdm2)
    # compute by subtracting the mean and then calculating cosine similarity
    vector1 = vector1 - np.mean(vector1, 1, keepdims=True)
    vector2 = vector2 - np.mean(vector2, 1, keepdims=True)
    sim = _cosine(vector1, vector2)
    return sim


def compare_cosine_cov_weighted(rdm1, rdm2, sigma_k=None):
    """calculates the cosine similarities between two RDMs objects

    Args:
        rdm1 (pyrsa.rdm.RDMs):
            first set of RDMs
        rdm2 (pyrsa.rdm.RDMs):
            second set of RDMs
    Returns:
        numpy.ndarray: dist:
            cosine similarities between the two RDMs

    """
    vector1, vector2 = _parse_input_rdms(rdm1, rdm2)
    sim = _cosine_cov_weighted(vector1, vector2, sigma_k)
    return sim


def compare_correlation_cov_weighted(rdm1, rdm2, sigma_k=None):
    """calculates the correlations between two RDMs objects after whitening
    with the covariance of the entries

    Args:
        rdm1 (pyrsa.rdm.RDMs):
            first set of RDMs
        rdm2 (pyrsa.rdm.RDMs):
            second set of RDMs
    Returns:
        numpy.ndarray: dist:
            correlations between the two RDMs

    """
    vector1, vector2 = _parse_input_rdms(rdm1, rdm2)
    # compute by subtracting the mean and then calculating cosine similarity
    vector1 = vector1 - np.mean(vector1, 1, keepdims=True)
    vector2 = vector2 - np.mean(vector2, 1, keepdims=True)
    sim = _cosine_cov_weighted(vector1, vector2, sigma_k)
    return sim


def compare_spearman(rdm1, rdm2):
    """calculates the spearman rank correlations between
    two RDMs objects

    Args:
        rdm1 (pyrsa.rdm.RDMs):
            first set of RDMs
        rdm2 (pyrsa.rdm.RDMs):
            second set of RDMs
    Returns:
        numpy.ndarray: dist:
            rank correlations between the two RDMs

    """
    vector1, vector2 = _parse_input_rdms(rdm1, rdm2)
    vector1 = np.apply_along_axis(scipy.stats.rankdata, 1, vector1)
    vector2 = np.apply_along_axis(scipy.stats.rankdata, 1, vector2)
    vector1 = vector1 - np.mean(vector1, 1, keepdims=True)
    vector2 = vector2 - np.mean(vector2, 1, keepdims=True)
    sim = _cosine(vector1, vector2)
    return sim


def compare_rho_a(rdm1, rdm2):
    """calculates the spearman rank correlations between
    two RDMs objects without tie correction

    Args:
        rdm1 (pyrsa.rdm.RDMs):
            first set of RDMs
        rdm2 (pyrsa.rdm.RDMs):
            second set of RDMs
    Returns:
        numpy.ndarray: dist:
            rank correlations between the two RDMs

    """
    vector1, vector2 = _parse_input_rdms(rdm1, rdm2)
    vector1 = np.apply_along_axis(scipy.stats.rankdata, 1, vector1)
    vector2 = np.apply_along_axis(scipy.stats.rankdata, 1, vector2)
    vector1 = vector1 - np.mean(vector1, 1, keepdims=True)
    vector2 = vector2 - np.mean(vector2, 1, keepdims=True)
    n = vector1.shape[1]
    sim = np.einsum('ij,kj->ik', vector1, vector2) / (n ** 3 - n) * 12
    return sim


def compare_kendall_tau(rdm1, rdm2):
    """calculates the Kendall-tau bs between two RDMs objects.
    Kendall-tau b is the version, which corrects for ties.
    We here use the implementation from scipy.

        Args:
            rdm1 (pyrsa.rdm.RDMs):
                first set of RDMs
            rdm2 (pyrsa.rdm.RDMs):
                second set of RDMs
        Returns:
            numpy.ndarray: dist:
                kendall-tau correlation between the two RDMs
    """
    vector1, vector2 = _parse_input_rdms(rdm1, rdm2)
    sim = _all_combinations(vector1, vector2, _kendall_tau)
    return sim


def compare_kendall_tau_a(rdm1, rdm2):
    """calculates the Kendall-tau a based distance between two RDMs objects.
    adequate when some models predict ties

        Args:
            rdm1 (pyrsa.rdm.RDMs):
                first set of RDMs
            rdm2 (pyrsa.rdm.RDMs):
                second set of RDMs
        Returns:
            numpy.ndarray: dist:
                kendall-tau a between the two RDMs
    """
    vector1, vector2 = _parse_input_rdms(rdm1, rdm2)
    sim = _all_combinations(vector1, vector2, _tau_a)
    return sim


def _all_combinations(vectors1, vectors2, func):
    """runs a function func on all combinations of v1 in vectors1
    and v2 in vectors2 and puts the results into an array

    Args:
        vectors1 (numpy.ndarray):
            first set of values
        vectors1 (numpy.ndarray):
            second set of values
        func (function):
            function to be applied, should take two input vectors
            and return one scalar
    Returns:
        numpy.ndarray: value: function result over all pairs

    """
    value = np.empty((len(vectors1), len(vectors2)))
    k1 = 0
    for v1 in vectors1:
        k2 = 0
        for v2 in vectors2:
            value[k1, k2] = func(v1, v2)
            k2 += 1
        k1 += 1
    return value


def _cosine_cov_weighted_slow(vector1, vector2, sigma_k=None):
    """computes the cosine similarities between two sets of vectors
    after whitening by their covariance.

    Args:
        vector1 (numpy.ndarray):
            first vectors (2D)
        vector1 (numpy.ndarray):
            second vectors (2D)
        sigma_k (Matrix):
            optional, covariance between pattern estimates

    Returns:
        cos (float):
            cosine of the angle between vectors

    """
    n_cond = _get_n_from_reduced_vectors(vector1.shape[1])
    v = _get_v(n_cond, sigma_k)
    # compute V^-1 vector1/2 for all vectors by solving Vx = vector1/2
    vector1_m = np.array([scipy.sparse.linalg.cg(v, vector1[i], atol=0)[0]
                          for i in range(vector1.shape[0])])
    vector2_m = np.array([scipy.sparse.linalg.cg(v, vector2[i], atol=0)[0]
                          for i in range(vector2.shape[0])])
    # compute the inner products v1^T (V^-1 v2) for all combinations
    cos = np.einsum('ij,kj->ik', vector1, vector2_m)
    # divide by sqrt(v1^T (V^-1 v1))
    cos /= np.sqrt(np.einsum('ij,ij->i', vector1,
                             vector1_m)).reshape((-1, 1))
    # divide by sqrt(v2^T (V^-1 v2))
    cos /= np.sqrt(np.einsum('ij,ij->i', vector2,
                             vector2_m)).reshape((1, -1))
    return cos


def _cosine_cov_weighted(vector1, vector2, sigma_k=None):
    """computes the cosine angles between two sets of vectors
    weighted by the covariance
    If no covariance is given this is computed using the linear CKA,
    which is equivalent in this case and faster to compute.
    Otherwise reverts to _cosine_cov_weighted_slow.

    Args:
        vector1 (numpy.ndarray):
            first vectors (2D)
        vector1 (numpy.ndarray):
            second vectors (2D)
        sigma_k (Matrix):
            optional, covariance between pattern estimates

    Returns:
        cos (float):
            cosine angle between vectors

    """
    if sigma_k is not None:
        cos = _cosine_cov_weighted_slow(vector1, vector2, sigma_k=sigma_k)
    else:
        # Compute the extended version of RDM vectors in whitened space
        vector1_m = _cov_weighting(vector1)
        vector2_m = _cov_weighting(vector2)
        # compute the inner products v1^T V^-1 v2 for all combinations
        cos = np.einsum('ij,kj->ik', vector1_m, vector2_m)
        # divide by sqrt(v1^T V^-1 v1)
        cos /= np.sqrt(np.einsum('ij,ij->i', vector1_m,
                                 vector1_m)).reshape((-1, 1))
        # divide by sqrt(v2^T V^-1 v2)
        cos /= np.sqrt(np.einsum('ij,ij->i', vector2_m,
                                 vector2_m)).reshape((1, -1))
    return cos


def _cov_weighting(vector):
    """Transforms a array of RDM vectors in to representation
    in which the elements are isotropic. This is a stretched-out
    second moment matrix, with the diagonal elements appended.
    To account for the fact that the off-diagonal elements are
    only there once, they are multipled by 2

    Args:
        vector (numpy.ndarray):
            RDM vectors (2D) N x n_dist

    Returns:
        vector_w:
            weighted vectors (M x n_dist + n_cond)

    """
    N, n_dist = vector.shape
    n_cond = _get_n_from_reduced_vectors(vector.shape[1])
    vector_w = -0.5 * np.c_[vector, np.zeros((N, n_cond))]
    rowI, colI = row_col_indicator_g(n_cond)
    sumI = rowI + colI
    m = vector_w @ sumI / n_cond  # Column and row means
    mm = np.sum(vector_w * 2, axis=1) / (n_cond * n_cond)  # Overall mean
    mm = mm.reshape(-1, 1)
    # subtract the column and row means and add overall mean
    vector_w = vector_w - m @ sumI.T + mm
    # Weight the off-diagnoal terms double
    vector_w[:, :n_dist] = vector_w[:, :n_dist] * np.sqrt(2)
    return vector_w


def _cosine(vector1, vector2):
    """computes the cosine angles between two sets of vectors

    Args:
        vector1 (numpy.ndarray):
            first vectors (2D)
        vector1 (numpy.ndarray):
            second vectors (2D)
    Returns:
        cos (float):
            cosine angle between vectors

    """
    # compute all inner products
    cos = np.einsum('ij,kj->ik', vector1, vector2)
    # divide by sqrt of the inner products with themselves
    cos /= np.sqrt(np.einsum('ij,ij->i', vector1, vector1)).reshape((-1, 1))
    cos /= np.sqrt(np.einsum('ij,ij->i', vector2, vector2)).reshape((1, -1))
    return cos


def _kendall_tau(vector1, vector2):
    """computes the kendall-tau between two vectors

    Args:
        vector1 (numpy.ndarray):
            first vector
        vector1 (numpy.ndarray):
            second vector
    Returns:
        tau (float):
            kendall-tau

    """
    tau = scipy.stats.kendalltau(vector1, vector2).correlation
    return tau


def _tau_a(vector1, vector2):
    """computes kendall-tau a between two vectors
    based on modifying scipy.stats.kendalltau

    Args:
        vector1 (numpy.ndarray):
            first vector
        vector1 (numpy.ndarray):
            second vector
    Returns:
        tau (float):
            kendall-tau a

    """
    size = vector1.size
    vector1, vector2 = _sort_and_rank(vector1, vector2)
    vector2, vector1 = _sort_and_rank(vector2, vector1)
    dis = _kendall_dis(vector1, vector2)  # discordant pairs
    obs = np.r_[True, (vector1[1:] != vector1[:-1]) |
                      (vector2[1:] != vector2[:-1]), True]
    cnt = np.diff(np.nonzero(obs)[0]).astype('int64', copy=False)
    ntie = (cnt * (cnt - 1) // 2).sum()  # joint ties
    xtie, x0, x1 = _count_rank_tie(vector1)     # ties in x, stats
    ytie, y0, y1 = _count_rank_tie(vector2)     # ties in y, stats
    tot = (size * (size - 1)) // 2
    # Note that tot = con + dis + (xtie - ntie) + (ytie - ntie) + ntie
    #               = con + dis + xtie + ytie - ntie
    con_minus_dis = tot - xtie - ytie + ntie - 2 * dis
    tau = con_minus_dis / tot
    # Limit range to fix computational errors
    tau = min(1., max(-1., tau))
    return tau


def _sort_and_rank(vector1, vector2):
    """does the sort and rank step of the _tau calculation"""
    perm = np.argsort(vector2, kind='mergesort')
    vector1 = vector1[perm]
    vector2 = vector2[perm]
    vector2 = np.r_[True, vector2[1:] != vector2[:-1]].cumsum(dtype=np.intp)
    return vector1, vector2


def _count_rank_tie(ranks):
    """ counts tied ranks for kendall-tau calculation"""
    cnt = np.bincount(ranks).astype('int64', copy=False)
    cnt = cnt[cnt > 1]
    return ((cnt * (cnt - 1) // 2).sum(),
            (cnt * (cnt - 1.) * (cnt - 2)).sum(),
            (cnt * (cnt - 1.) * (2*cnt + 5)).sum())


def _get_v(n_cond, sigma_k):
    """ get the rdm covariance from sigma_k """
    # calculate Xi
    c_mat = pairwise_contrast_sparse(np.arange(n_cond))
    if sigma_k is None:
        xi = c_mat @ c_mat.transpose()
    else:
        sigma_k = scipy.sparse.csr_matrix(sigma_k)
        xi = c_mat @ sigma_k @ c_mat.transpose()
    # calculate V
    v = xi.multiply(xi).tocsc()
    return v


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
    vector1_no_nan = vector1[~np.isnan(vector1)].reshape(vector1.shape[0], -1)
    vector2_no_nan = vector2[~np.isnan(vector2)].reshape(vector2.shape[0], -1)
    if not vector1_no_nan.shape[1] == vector2_no_nan.shape[1]:
        raise ValueError('rdm1 and rdm2 have different nan positions')
    return vector1_no_nan, vector2_no_nan
