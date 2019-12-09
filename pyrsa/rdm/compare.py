#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 13:01:12 2019

@author: heiko
"""
import numpy as np
import scipy.stats


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
    dist = _average_all_combinations(vector1, vector2, _cosine)
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
    dist = _average_all_combinations(vector1, vector2, _cosine)
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
                correlation distance between the two RDMs
    """
    vector1, vector2 = _parse_input_rdms(rdm1, rdm2)
    dist = _average_all_combinations(vector1, vector2, scipy.stats.spearmanr)
    return 1 - dist


def _average_all_combinations(vectors1, vectors2, func):
    """
    runs a function func on all combinations of v1 in vectors1
    and v2 in vectors2 and averages the results

        Args:
            vectors1 (numpy.ndarray):
                first set of values
            vectors1 (numpy.ndarray):
                second set of values
            func (function):
                function to be applied, should take two input vectors
                and return one scalar
        Returns:
            value (float):
                average function result over all pairs
    """
    sum_val = 0
    for v1 in vectors1:
        for v2 in vectors2:
            sum_val += func(v1, v2)
    return sum_val / vectors1.shape[0] / vectors2.shape[0]


def _cosine(vector1, vector2):
    """
    computes the cosine angle between two vectors

        Args:
            vector1 (numpy.ndarray):
                first vector
            vector1 (numpy.ndarray):
                second vector
        Returns:
            cos (float):
                cosine angle between angles
    """
    cos = (np.sum(vector1 * vector2) /
           np.sqrt(np.sum(vector1 * vector1)) /
           np.sqrt(np.sum(vector2 * vector2)))
    return cos


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
