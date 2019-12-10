#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 13:01:12 2019

@author: heiko
"""
import numpy as np
import scipy.stats


def compare(rdm1, rdm2, method='cosine'):
    """
    calculates a distance between two RDMs objects

        Args:
            rdm1 (pyrsa.rdm.RDMs):
                first set of RDMs
            rdm2 (pyrsa.rdm.RDMs):
                second set of RDMs
            method (string):
                which method to use, options are:
                'cosine' = cosine distance
                'spearman' = spearman rank correlation distance
                'corr' = pearson correlation distance
                'kendall' = kendall-tau based distance
        Returns:
            dist (float):
                dissimilarity between the two RDMs
    """
    if method == 'cosine':
        dist = compare_cosine(rdm1, rdm2)
    elif method == 'spearman':
        dist = compare_rank_corr(rdm1, rdm2)
    elif method == 'corr':
        dist = compare_correlation(rdm1, rdm2)
    elif method == 'kendall':
        dist = compare_kendall_tau(rdm1, rdm2)
    else:
        raise ValueError('Unknown RDM comparison method requested!')
    return dist


def compare_cosine(rdm1, rdm2):
    """
    calculates the cosine distance between two RDMs objects

        Args:
            rdm1 (pyrsa.rdm.RDMs):
                first set of RDMs
            rdm2 (pyrsa.rdm.RDMs):
                second set of RDMs
        Returns:
            dist (float):
                cosine distance between the two RDMs
    """
    vector1, vector2 = _parse_input_rdms(rdm1, rdm2)
    dist = _cosine(vector1, vector2)
    return 1 - dist


def compare_correlation(rdm1, rdm2):
    """
    calculates the correlation distance between two RDMs objects

        Args:
            rdm1 (pyrsa.rdm.RDMs):
                first set of RDMs
            rdm2 (pyrsa.rdm.RDMs):
                second set of RDMs
        Returns:
            dist (float):
                correlation distance between the two RDMs
    """
    vector1, vector2 = _parse_input_rdms(rdm1, rdm2)
    vector1 = vector1 - np.mean(vector1, 1, keepdims=True)
    vector2 = vector2 - np.mean(vector2, 1, keepdims=True)
    dist = _cosine(vector1, vector2)
    return 1 - dist


def compare_rank_corr(rdm1, rdm2):
    """
    calculates the correlation distance between two RDMs objects

        Args:
            rdm1 (pyrsa.rdm.RDMs):
                first set of RDMs
            rdm2 (pyrsa.rdm.RDMs):
                second set of RDMs
        Returns:
            dist (float):
                rank correlation distance between the two RDMs
    """
    vector1, vector2 = _parse_input_rdms(rdm1, rdm2)
    dist = _all_combinations(vector1, vector2, _spearman_r)
    return 1 - dist


def compare_kendall_tau(rdm1, rdm2):
    """
    calculates the Kendall-tau based distance between two RDMs objects

        Args:
            rdm1 (pyrsa.rdm.RDMs):
                first set of RDMs
            rdm2 (pyrsa.rdm.RDMs):
                second set of RDMs
        Returns:
            dist (float):
                kendall-tau based distance between the two RDMs
    """
    vector1, vector2 = _parse_input_rdms(rdm1, rdm2)
    dist = _all_combinations(vector1, vector2, _kendall_tau)
    return 1 - dist


def _all_combinations(vectors1, vectors2, func):
    """
    runs a function func on all combinations of v1 in vectors1
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
            value (numpy.ndarray):
                function result over all pairs
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


def _cosine(vector1, vector2):
    """
    computes the cosine angles between two sets of vectors

        Args:
            vector1 (numpy.ndarray):
                first vectors (2D)
            vector1 (numpy.ndarray):
                second vectors (2D)
        Returns:
            cos (float):
                cosine angle between angles
    """
    cos = np.einsum('ij,kj->ik', vector1, vector2)
    cos /= np.sqrt(np.einsum('ij,ij->i', vector1, vector1)).reshape((-1,1))
    cos /= np.sqrt(np.einsum('ij,ij->i', vector2, vector2)).reshape((1,-1))
    return cos


def _kendall_tau(vector1, vector2):
    """
    computes the kendall-tau between two vectors

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


def _spearman_r(vector1, vector2):
    """
    computes the spearman rank correlation between two vectors

        Args:
            vector1 (numpy.ndarray):
                first vector
            vector1 (numpy.ndarray):
                second vector
        Returns:
            corr (float):
                spearman r
    """
    corr = scipy.stats.spearmanr(vector1, vector2).correlation
    return corr


def _parse_input_rdms(rdm1, rdm2):
    """
    Gets the vector representation of input RDMs, raises an error if
    the two RDMs objects have different dimensions

        Args:
            rdm1 (pyrsa.rdm.RDMs):
                first set of RDMs
            rdm2 (pyrsa.rdm.RDMs):
                second set of RDMs
    """
    vector1 = rdm1.get_vectors()
    vector2 = rdm2.get_vectors()
    if not vector1.shape[1] == vector2.shape[1]:
        raise ValueError('rdm1 and rdm2 must be RDMs of equal shape')
    return vector1, vector2
