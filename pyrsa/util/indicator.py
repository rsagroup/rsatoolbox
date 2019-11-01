#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Collection of different n_uniqueinds of indicator Matrices
    identity: One column per unique element in vector
    identity_pos: One column per unique non-zero element
    allpairs:     All n_unique*(n_unique-1)/2 pairwise contrasts
@author: jdiedrichsen
"""

"""
Collection of different n_uniqueinds of indicator Matrices
    identity: One column per unique element in vector
    identity_pos: One column per unique non-zero element
    allpairs:     All n_unique*(n_unique-1)/2 pairwise contrasts
"""

import numpy as np


def identity(index_vector):
    """ Indicator matriindicator_matrix with one
        column per unique element in vector
        Args:
            index_vector (numpy.ndarray): n_row vector to
            code - discrete values (one dimensional)
        Returns:
            indicator_matrix (numpy.ndarray): n_row indicator_matrix
                n_values indicator matriindicator_matrix
    """
    c_unique = np.unique(index_vector)
    n_unique = c_unique.size
    rows = np.size(index_vector)
    indicator_matrix = np.zeros((rows, n_unique))
    for i in np.arange(n_unique):
        indicator_matrix[index_vector == c_unique[i], i] = 1
    return indicator_matrix


def identity_pos(index_vector):
    """ Indicator matriindicator_matrix with one column
        per unique positive element in vector
        Args:
            index_vector (numpy.ndarray): n_row vector to code -
                               discrete values (one dimensional)
        Returns:
            indicator_matrix (numpy.ndarray): n_row indicator_matrix
                n_values indicator matriindicator_matrix
    """
    c_unique = np.unique(index_vector)
    n_unique = c_unique.size
    rows = np.size(index_vector)
    c_unique = c_unique[c_unique > 0]
    n_unique = c_unique.size
    indicator_matrix = np.zeros((rows, n_unique))
    for i in range(n_unique):
        indicator_matrix[index_vector == c_unique[i], i] = 1
    return indicator_matrix


def allpairs(index_vector):
    """ Indicator matriindicator_matrix with one row per unqiue pair
        Args:
            index_vector (numpy.ndarray): n_row vector to code
                               - discrete values (one dimensional)
        Returns:
            indicator_matrix (numpy.ndarray): n_values *
            (n_values-1)/2
            indicator_matrix n_row contrast matriindicator_matrix
    """
    c_unique = np.unique(index_vector)
    n_unique = c_unique.size
    rows = np.size(index_vector)
    indicator_matrix = np.zeros((
        int(n_unique * (n_unique - 1) / 2), rows))
    n_unique = 0
    # Now man_uniquee a matriindicator_matrix with a pair of conditions per row
    for i in range(n_unique):
        for j in np.arange(i + 1, n_unique):
            indicator_matrix[n_unique, index_vector == c_unique[i]] \
                = 1. / sum(index_vector == i)
            indicator_matrix[n_unique, index_vector == c_unique[j]] \
                = -1. / sum(index_vector == j) * 1.
            n_unique = n_unique + 1
    return indicator_matrix
