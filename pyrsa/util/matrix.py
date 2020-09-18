#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Collection of different utility Matrices
"""

import numpy as np
from scipy.sparse import coo_matrix


def indicator(index_vector, positive=False):
    """ Indicator matrix with one
    column per unique element in vector

    Args:
        index_vector (numpy.ndarray): n_row vector to
            code - discrete values (one dimensional)
        positive (bool): should the function ignore zero
            negative entries in the index_vector?
            Default: false

    Returns:
        indicator_matrix (numpy.ndarray): nrow x nconditions
            indicator matrix

    """
    c_unique = np.unique(index_vector)
    n_unique = c_unique.size
    rows = np.size(index_vector)
    if positive:
        c_unique = c_unique[c_unique > 0]
        n_unique = c_unique.size
    indicator_matrix = np.zeros((rows, n_unique))
    for i in np.arange(n_unique):
        indicator_matrix[index_vector == c_unique[i], i] = 1
    return indicator_matrix


def pairwise_contrast(index_vector):
    """ Contrast matrix with one row per unqiue pairwise contrast

    Args:
        index_vector (numpy.ndarray): n_row vector to code
            discrete values (one dimensional)

    Returns:
        numpy.ndarray: indicator_matrix: n_values * (n_values-1)/2 x n_row
        contrast matrix

    """
    c_unique = np.unique(index_vector)
    n_unique = c_unique.size
    rows = np.size(index_vector)
    cols = int(n_unique * (n_unique - 1) / 2)
    indicator_matrix = np.zeros((cols, rows))
    n_row = 0
    # Now make an indicator_matrix with a pair of conditions per row
    for i in range(n_unique):
        for j in np.arange(i + 1, n_unique):
            select = (index_vector == c_unique[i])
            indicator_matrix[n_row, select] = 1. / np.sum(select)
            select = (index_vector == c_unique[j])
            indicator_matrix[n_row, select] = -1. / np.sum(select)
            n_row = n_row + 1
    return indicator_matrix


def pairwise_contrast_sparse(index_vector):
    """ Contrast matrix with one row per unqiue pairwise contrast

    Args:
        index_vector (numpy.ndarray): n_row vector to code
            discrete values (one dimensional)

    Returns:
        scipy.sparse.csr_matrix: indicator_matrix:
            n_values * (n_values-1)/2 x n_row contrast matrix

    """
    c_unique = np.unique(index_vector)
    n_unique = c_unique.size
    rows = np.size(index_vector)
    cols = int(n_unique * (n_unique - 1) / 2)
    # Now make an indicator_matrix with a pair of conditions per row
    n_repeats = np.zeros(n_unique, dtype=int)
    select = [None] * n_unique
    for i in range(n_unique):
        sel = (index_vector == c_unique[i])
        n_repeats[i] = np.sum(sel)
        select[i] = list(np.where(index_vector == c_unique[i])[0])
    n_row = 0
    dat = []
    idx_i = []
    idx_j = []
    for i in range(n_unique):
        for j in np.arange(i + 1, n_unique):
            dat += [1/n_repeats[i]] * n_repeats[i]
            idx_i += [n_row] * n_repeats[i]
            idx_j += select[i]
            dat += [-1/n_repeats[j]] * n_repeats[j]
            idx_i += [n_row] * n_repeats[j]
            idx_j += select[j]
            n_row = n_row + 1
    indicator_matrix = coo_matrix((dat, (idx_i, idx_j)),
                                  shape=(cols, rows))
    return indicator_matrix.asformat("csr")


def centering(size):
    """ generates a centering matrix

    Args:
        size (int): size of the center matrix

    Returns:
        centering_matrix (numpy.ndarray): size * size
    """
    centering_matrix = np.identity(size) - np.ones(size) / size
    return centering_matrix


def row_col_indicator_rdm(n_cond):
    """ generates a row and column indicator matrix for an RDM vector

    Args:
        n_cond (int): Number of conditions underlying the RDM

    Returns:
        row_indicator (numpy.ndarray): n_cond (n_cond-1)/2 * n_cond
        col_indicator (numpy.ndarray): n_cond (n_cond-1)/2 * n_cond
    """
    n_dist = int(n_cond * (n_cond - 1) / 2)
    row_i = np.zeros((n_dist, n_cond))
    col_i = np.zeros((n_dist, n_cond))
    _row_col_indicator(row_i, col_i, n_cond)
    return (row_i, col_i)


def row_col_indicator_g(n_cond):
    """ generates a row and column indicator matrix for a vectorized
    second moment matrix. The vectorized version has the off-diagonal elements
    first (like in an RDM), and then appends the diagnoal.
    You can vectorize a second momement matrix G by
    np.diag(row_i@G@col_i.T) =  np.sum(col_i*(row_i@G)),axis=1)

    Args:
        n_cond (int): Number of conditions underlying the second moment

    Returns:
        row_indicator (numpy.ndarray): n_cond (n_cond-1)/2+n_cond * n_cond
        col_indicator (numpy.ndarray): n_cond (n_cond-1)/2+n_cond * n_cond
    """
    n_elem = int(n_cond * (n_cond - 1) / 2)+n_cond  # Number of elements in G
    row_i = np.zeros((n_elem, n_cond))
    col_i = np.zeros((n_elem, n_cond))
    _row_col_indicator(row_i, col_i, n_cond)
    np.fill_diagonal(row_i[-n_cond:, :], 1)
    np.fill_diagonal(col_i[-n_cond:, :], 1)
    return (row_i, col_i)


def _row_col_indicator(row_i, col_i, n_cond):
    """ Helper function that writes the correct pattern for the
    row / column indicator matrix

    Args:
        row_indicator: row_i (numpy.ndarray)
        col_indicator: row_i (numpy.ndarray)
        n_cond (int): Number of conditions underlying the second moment
    """
    j = 0
    for i in range(n_cond):
        row_i[j:j + n_cond - i - 1, i] = 1
        np.fill_diagonal(col_i[j:j + n_cond - i - 1, i + 1:], 1)
        j = j + (n_cond - i - 1)
