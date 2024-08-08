#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comparison methods for comparing two RDMs objects
"""
import numpy as np
import scipy.stats
from scipy import linalg
from scipy.optimize import minimize
from scipy.stats._stats import _kendall_dis
from scipy.spatial.distance import squareform
from rsatoolbox.util.matrix import pairwise_contrast_sparse
from rsatoolbox.util.matrix import pairwise_contrast
from rsatoolbox.util.rdm_utils import _get_n_from_reduced_vectors
from rsatoolbox.util.rdm_utils import _get_n_from_length
from rsatoolbox.util.matrix import row_col_indicator_g
from rsatoolbox.util.rdm_utils import batch_to_matrices


def compare(rdm1, rdm2, method='cosine', sigma_k=None):
    """calculates the similarity between two RDMs objects using a chosen method

    Args:
        rdm1 (rsatoolbox.rdm.RDMs):
            first set of RDMs

        rdm2 (rsatoolbox.rdm.RDMs):
            second set of RDMs

        method (string): which method to use, options are:

            'cosine' = cosine similarity

            'spearman' = spearman rank correlation

            'corr' = pearson correlation

            'kendall' = kendall-tau b

            'tau-a' = kendall-tau a

            'rho-a' = spearman correlation without tie correction

            'corr_cov' = pearson correlation after whitening

            'cosine_cov' = unbiased distance correlation
            which is equivalent to the cosine dinstance after whitening

            'neg_riem_dist' = negative riemannian distance

            'bures' = bures similarity of equivalend cented kernel matrices

            'bures_metric' = distances based on bures similarity, which is a metric

        sigma_k (numpy.ndarray):
            covariance matrix of the pattern estimates.
            Used only for methods 'corr_cov' and 'cosine_cov'.

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
    elif method in ('kendall', 'tau-b'):
        sim = compare_kendall_tau(rdm1, rdm2)
    elif method == 'tau-a':
        sim = compare_kendall_tau_a(rdm1, rdm2)
    elif method == 'rho-a':
        sim = compare_rho_a(rdm1, rdm2)
    elif method == 'corr_cov':
        sim = compare_correlation_cov_weighted(rdm1, rdm2, sigma_k=sigma_k)
    elif method == 'cosine_cov':
        sim = compare_cosine_cov_weighted(rdm1, rdm2, sigma_k=sigma_k)
    elif method == 'neg_riem_dist':
        sim = compare_neg_riemannian_distance(rdm1, rdm2, sigma_k=sigma_k)
    elif method == 'bures':
        sim = compare_bures_similarity(rdm1, rdm2)
    elif method == 'bures_metric':
        sim = compare_bures_metric(rdm1, rdm2)
    else:
        raise ValueError('Unknown RDM comparison method requested!')
    return sim


def compare_cosine(rdm1, rdm2):
    """calculates the cosine similarities between two RDMs objects

    Args:
        rdm1 (rsatoolbox.rdm.RDMs):
            first set of RDMs
        rdm2 (rsatoolbox.rdm.RDMs):
            second set of RDMs
    Returns:
        numpy.ndarray: dist
            cosine similarity between the two RDMs

    """
    vector1, vector2, _ = _parse_input_rdms(rdm1, rdm2)
    sim = _cosine(vector1, vector2)
    return sim


def compare_correlation(rdm1, rdm2):
    """calculates the correlations between two RDMs objects

    Args:
        rdm1 (rsatoolbox.rdm.RDMs):
            first set of RDMs
        rdm2 (rsatoolbox.rdm.RDMs):
            second set of RDMs
    Returns:
        numpy.ndarray: dist:
            correlations between the two RDMs

    """
    vector1, vector2, _ = _parse_input_rdms(rdm1, rdm2)
    # compute by subtracting the mean and then calculating cosine similarity
    vector1 = vector1 - np.mean(vector1, 1, keepdims=True)
    vector2 = vector2 - np.mean(vector2, 1, keepdims=True)
    sim = _cosine(vector1, vector2)
    return sim


def compare_cosine_cov_weighted(rdm1, rdm2, sigma_k=None):
    """calculates the cosine similarities between two RDMs objects

    Args:
        rdm1 (rsatoolbox.rdm.RDMs):
            first set of RDMs
        rdm2 (rsatoolbox.rdm.RDMs):
            second set of RDMs
    Returns:
        numpy.ndarray: dist:
            cosine similarities between the two RDMs

    """
    vector1, vector2, nan_idx = _parse_input_rdms(rdm1, rdm2)
    sim = _cosine_cov_weighted(vector1, vector2, sigma_k, nan_idx)
    return sim


def compare_correlation_cov_weighted(rdm1, rdm2, sigma_k=None):
    """calculates the correlations between two RDMs objects after whitening
    with the covariance of the entries

    Args:
        rdm1 (rsatoolbox.rdm.RDMs):
            first set of RDMs
        rdm2 (rsatoolbox.rdm.RDMs):
            second set of RDMs

    Returns:
        numpy.ndarray: dist:
            correlations between the two RDMs

    """
    vector1, vector2, nan_idx = _parse_input_rdms(rdm1, rdm2)
    # compute by subtracting the mean and then calculating cosine similarity
    vector1 = vector1 - np.mean(vector1, 1, keepdims=True)
    vector2 = vector2 - np.mean(vector2, 1, keepdims=True)
    sim = _cosine_cov_weighted(vector1, vector2, sigma_k, nan_idx)
    return sim


def compare_spearman(rdm1, rdm2):
    """calculates the spearman rank correlations between
    two RDMs objects

    Args:
        rdm1 (rsatoolbox.rdm.RDMs):
            first set of RDMs
        rdm2 (rsatoolbox.rdm.RDMs):
            second set of RDMs
    Returns:
        numpy.ndarray: dist:
            rank correlations between the two RDMs

    """
    vector1, vector2, _ = _parse_input_rdms(rdm1, rdm2)
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
        rdm1 (rsatoolbox.rdm.RDMs):
            first set of RDMs
        rdm2 (rsatoolbox.rdm.RDMs):
            second set of RDMs
    Returns:
        numpy.ndarray: dist:
            rank correlations between the two RDMs

    """
    vector1, vector2, _ = _parse_input_rdms(rdm1, rdm2)
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
        rdm1 (rsatoolbox.rdm.RDMs):
            first set of RDMs
        rdm2 (rsatoolbox.rdm.RDMs):
            second set of RDMs
    Returns:
        numpy.ndarray: dist:
            kendall-tau correlation between the two RDMs
    """
    vector1, vector2, _ = _parse_input_rdms(rdm1, rdm2)
    sim = _all_combinations(vector1, vector2, _kendall_tau)
    return sim


def compare_kendall_tau_a(rdm1, rdm2):
    """calculates the Kendall-tau a based distance between two RDMs objects.
    adequate when some models predict ties

    Args:
        rdm1 (rsatoolbox.rdm.RDMs):
            first set of RDMs
        rdm2 (rsatoolbox.rdm.RDMs):
            second set of RDMs
    Returns:
        numpy.ndarray: dist:
            kendall-tau a between the two RDMs
    """
    vector1, vector2, _ = _parse_input_rdms(rdm1, rdm2)
    sim = _all_combinations(vector1, vector2, _tau_a)
    return sim


def compare_neg_riemannian_distance(rdm1, rdm2, sigma_k=None):
    """calculates the negative Riemannian distance between two RDMs objects.

    Args:
        rdm1 (rsatoolbox.rdm.RDMs):
            first set of RDMs
        rdm2 (rsatoolbox.rdm.RDMs):
            second set of RDMs
    Returns:
        numpy.ndarray: dist:
            negative Riemannian distance between the two RDMs
    """
    vector1, vector2, _ = _parse_input_rdms(rdm1, rdm2)
    n_cond = _get_n_from_length(vector1.shape[1])
    if sigma_k is None:
        sigma_k = np.eye(n_cond)
    P = np.block([-1*np.ones((n_cond - 1, 1)), np.eye(n_cond - 1)])
    sigma_k_hat = P@sigma_k@P.T
    # construct RDM to 2nd-moment (G) transformation
    pairs = pairwise_contrast(np.arange(n_cond-1))
    pairs[pairs == -1] = 1
    T = np.block([
        [np.eye(n_cond - 1), np.zeros((n_cond-1, vector1.shape[1] - n_cond + 1))],
        [0.5 * pairs, np.diag(-0.5 * np.ones(vector1.shape[1] - n_cond + 1))]])
    vec_G1 = vector1@np.transpose(T)
    vec_G2 = vector2@np.transpose(T)

    sim = _all_combinations(vec_G1, vec_G2, _riemannian_distance, sigma_k_hat)
    return sim


def compare_bures_similarity(rdm1, rdm2):
    """calculates the Bures similarity between two RDMs objects.

    Args:
        rdm1 (rsatoolbox.rdm.RDMs):
            first set of RDMs
        rdm2 (rsatoolbox.rdm.RDMs):
            second set of RDMs
    Returns:
        numpy.ndarray: dist:
            Bures similarity between the two RDMs
    """
    vector1, vector2, _ = _parse_input_rdms(rdm1, rdm2)
    G1, _, _ = batch_to_matrices(-vector1 / 2)
    G2, _, _ = batch_to_matrices(-vector2 / 2)
    s1 = np.mean(G1, 1, keepdims=True)
    G1 = G1 - s1 - np.transpose(s1, (0, 2, 1)) + np.mean(s1, 2, keepdims=True)
    s2 = np.mean(G2, 1, keepdims=True)
    G2 = G2 - s2 - np.transpose(s2, (0, 2, 1)) + np.mean(s2, 2, keepdims=True)
    sim = _all_combinations(G1, G2, _bures_similarity_first_way)
    return sim


def compare_bures_metric(rdm1, rdm2):
    """calculates the squared Bures metric between two RDMs objects.

    Args:
        rdm1 (rsatoolbox.rdm.RDMs):
            first set of RDMs
        rdm2 (rsatoolbox.rdm.RDMs):
            second set of RDMs
    Returns:
        numpy.ndarray: dist:
            squared Bures metric between the two RDMs
    """
    vector1, vector2, _ = _parse_input_rdms(rdm1, rdm2)
    G1, _, _ = batch_to_matrices(-vector1 / 2)
    G2, _, _ = batch_to_matrices(-vector2 / 2)
    s1 = np.mean(G1, 1, keepdims=True)
    G1 = G1 - s1 - np.transpose(s1, (0, 2, 1)) + np.mean(s1, 2, keepdims=True)
    s2 = np.mean(G2, 1, keepdims=True)
    G2 = G2 - s2 - np.transpose(s2, (0, 2, 1)) + np.mean(s2, 2, keepdims=True)
    sim = _all_combinations(G1, G2, _sq_bures_metric_first_way)
    return sim


def _all_combinations(vectors1, vectors2, func, *args, **kwargs):
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
    for k1, v1 in enumerate(vectors1):
        for k2, v2 in enumerate(vectors2):
            value[k1, k2] = func(v1, v2, *args, **kwargs)
    return value


def _cosine_cov_weighted_slow(vector1, vector2, sigma_k=None, nan_idx=None):
    """computes the cosine similarities between two sets of vectors
    after whitening by their covariance.

    Args:
        vector1 (numpy.ndarray):
            first vectors (2D)
        vector1 (numpy.ndarray):
            second vectors (2D)
        sigma_k (Matrix):
            optional, covariance between pattern estimates
        nan_idx (numpy.ndarray):
            vector of non-nan entries from input parsing

    Returns:
        cos (float):
            cosine of the angle between vectors

    """
    if nan_idx is not None:
        n_cond = _get_n_from_reduced_vectors(nan_idx.reshape(1, -1))
        v = _get_v(n_cond, sigma_k)
        v = v[nan_idx][:, nan_idx]
    else:
        n_cond = _get_n_from_reduced_vectors(vector1)
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


def _cosine_cov_weighted(vector1, vector2, sigma_k=None, nan_idx=None):
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
    if (sigma_k is not None) and (sigma_k.ndim >= 2):
        cos = _cosine_cov_weighted_slow(
            vector1, vector2, sigma_k=sigma_k, nan_idx=nan_idx)
    else:
        if nan_idx is None:
            nan_idx = np.ones(vector1[0].shape, bool)
        # Compute the extended version of RDM vectors in whitened space
        vector1_m = _cov_weighting(vector1, nan_idx, sigma_k)
        vector2_m = _cov_weighting(vector2, nan_idx, sigma_k)
        cos = _cosine(vector1_m, vector2_m)
    return cos


def _cov_weighting(vector, nan_idx, sigma_k=None):
    """Transforms an array of RDM vectors in to representation
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
    n_cond = _get_n_from_length(nan_idx.shape[0])
    vector_w = -0.5 * np.c_[vector, np.zeros((N, n_cond))]
    rowI, colI = row_col_indicator_g(n_cond)
    sumI = rowI + colI
    if np.all(nan_idx):
        # column and row means
        m = vector_w @ sumI / n_cond
        # Overall mean
        mm = np.sum(vector_w * 2, axis=1, keepdims=True) / (n_cond * n_cond)
        # subtract the column and row means and add overall mean
        vector_w = vector_w - m @ sumI.T + mm
        if sigma_k is not None:
            if sigma_k.ndim == 1:
                sigma_k_sqrt = np.sqrt(sigma_k)
                vector_w /= rowI @ sigma_k_sqrt
                vector_w /= colI @ sigma_k_sqrt
            elif sigma_k.ndim == 2:
                l_sigma_k = np.linalg.inv(np.linalg.cholesky(sigma_k))
                Gs = np.empty((vector.shape[0], n_cond, n_cond))
                for i_vec in range(vector.shape[0]):
                    G = scipy.spatial.distance.squareform(
                        vector_w[i_vec, :n_dist])
                    np.fill_diagonal(G, vector_w[i_vec, n_dist:])
                    Gs[i_vec] = G
                # These two are the slow lines for this whitening
                Gs = np.einsum('ij,mjk,lk->mil', l_sigma_k, Gs, l_sigma_k)
                vector_w = np.einsum('ij,mjk,ik->mi', rowI, Gs, colI)
    else:
        nan_idx_ext = np.concatenate((nan_idx, np.ones(n_cond, bool)))
        sumI = sumI[nan_idx_ext]
        # get matrix for double centering with missing values:
        sumI[n_dist:, :] /= 2
        diag = np.concatenate((np.ones((n_dist, 1)) / 2, np.ones((n_cond, 1))))
        # one line version much faster here!
        vector_w = vector_w - (
            vector_w
            @ sumI @ np.linalg.inv(sumI.T @ (diag * sumI)) @ (diag * sumI).T)
        if sigma_k is not None:
            if sigma_k.ndim == 1:
                sigma_k_sqrt = np.sqrt(sigma_k)
                vector_w /= rowI[nan_idx_ext] @ sigma_k_sqrt
                vector_w /= colI[nan_idx_ext] @ sigma_k_sqrt
            elif sigma_k.ndim == 2:
                raise ValueError('cannot handle sigma_k and nans')
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
    norm_1 = np.sqrt(np.einsum('ij,ij->i', vector1, vector1))
    norm_2 = np.sqrt(np.einsum('ij,ij->i', vector2, vector2))
    sel_1 = norm_1 > 0
    sel_2 = norm_2 > 0
    # without more indexing if all vectors are nonzero length
    if np.all(sel_1) and np.all(sel_2):
        # compute all inner products
        cos_ok = np.einsum('ij,kj->ik', vector1, vector2)
        # divide by sqrt of the inner products with themselves
        cos_ok /= norm_1.reshape((-1, 1))
        cos_ok /= norm_2.reshape((1, -1))
        return cos_ok
    # keep track of indexing if some vectors are 0
    # compute all inner products
    cos_ok = np.einsum('ij,kj->ik', vector1[sel_1], vector2[sel_2])
    # divide by sqrt of the inner products with themselves
    cos_ok /= norm_1[sel_1].reshape((-1, 1))
    cos_ok /= norm_2[sel_2].reshape((1, -1))
    cos = np.zeros((vector1.shape[0], vector2.shape[0]))
    np.putmask(cos, np.outer(norm_1 > 0, norm_2 > 0), cos_ok)
    return cos


def _riemannian_distance(vec_G1, vec_G2, sigma_k):
    """computes the Riemannian distance between two vectorized second moments

    Args:
        vec_G1 (numpy.ndarray):
            first vectorized second-moment
        vec_G2 (numpy.ndarray):
            second vectorized second-moment

        Returns:
            neg_riem (float):
                negative riemannian distance
    """
    n_cond = _get_n_from_length(len(vec_G1))
    G1 = np.diag(vec_G1[0:(n_cond-1)])+squareform(vec_G1[(n_cond-1):len(vec_G1)])
    G2 = np.diag(vec_G2[0:(n_cond-1)])+squareform(vec_G2[(n_cond-1):len(vec_G2)])

    def fun(theta):
        return np.sqrt((np.log(linalg.eigvalsh(
            np.exp(theta[0]) * G1 + np.exp(theta[1]) * sigma_k, G2))**2).sum())
    theta = minimize(fun, (0, 0), method='Nelder-Mead')
    neg_riem = -1 * theta.fun
    return neg_riem


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
    xtie, _, _ = _count_rank_tie(vector1)     # ties in x, stats
    ytie, _, _ = _count_rank_tie(vector2)     # ties in y, stats
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
    elif sigma_k.ndim == 1:
        sigma_k = scipy.sparse.diags(sigma_k)
        xi = c_mat @ sigma_k @ c_mat.transpose()
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
        rdm1 (rsatoolbox.rdm.RDMs):
            first set of RDMs
        rdm2 (rsatoolbox.rdm.RDMs):
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
    return vector1_no_nan, vector2_no_nan, nan_idx[0]


def _sq_bures_metric_first_way(A, B):
    va, ua = np.linalg.eigh(A)
    Asq = ua @ (np.sqrt(np.maximum(va[:, None], 0.0)) * ua.T)
    return (
        np.trace(A) + np.trace(B)
        - 2 * np.sum(np.sqrt(np.maximum(0.0, np.linalg.eigvalsh(Asq @ B @ Asq))))
    )


def _sq_bures_metric_second_way(A, B):
    va, ua = np.linalg.eigh(A)
    vb, ub = np.linalg.eigh(B)
    sva = np.sqrt(np.maximum(va, 0.0))
    svb = np.sqrt(np.maximum(vb, 0.0))
    return (
        np.sum(va) + np.sum(vb) - 2 * np.sum(
            np.linalg.svd(
                (sva[:, None] * ua.T) @ (ub * svb[None, :]),
                compute_uv=False
            )
        )
    )


def _bures_similarity_first_way(A, B):
    va, ua = np.linalg.eigh(A)
    Asq = ua @ (np.sqrt(np.maximum(va[:, None], 0.0)) * ua.T)
    num = np.sum(np.sqrt(np.maximum(np.linalg.eigvalsh(Asq @ B @ Asq), 0.0)))
    denom = np.sqrt(np.trace(A) * np.trace(B))
    return num / denom


def _bures_similarity_second_way(A, B):
    va, ua = np.linalg.eigh(A)
    vb, ub = np.linalg.eigh(B)
    sva = np.sqrt(np.maximum(va, 0.0))
    svb = np.sqrt(np.maximum(vb, 0.0))
    num = np.sum(
        np.linalg.svd(
            (sva[:, None] * ua.T) @ (ub * svb[None, :]),
            compute_uv=False
        )
    )
    denom = np.sqrt(np.sum(va) * np.sum(vb))
    return num / denom
