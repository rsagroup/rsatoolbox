#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Collection of different utility Matrices
    indicator:  indicator variable for each unique element in vector
    pairwise_contrast:  All n_unique*(n_unique-1)/2 pairwise contrasts
    centering: Centering matrix which removes the column or row mean
@author: jdiedrichsen
"""

import numpy as np


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
                               - discrete values (one dimensional)
        Returns:
            constrast_matrix (numpy.ndarray): n_values *
                (n_values-1)/2 
    """
    c_unique = np.unique(index_vector)
    n_unique = c_unique.size
    rows = np.size(index_vector)
    indicator_matrix = np.zeros((
        int(n_unique * (n_unique - 1) / 2), rows))
    n_row = 0
    # Now make an indicator_matrix with a pair of conditions per row
    for i in range(n_unique):
        for j in np.arange(i + 1, n_unique):
            select = (index_vector == c_unique[i])
            indicator_matrix[n_row, select] = 1. / sum(select)
            select = (index_vector == c_unique[j])
            indicator_matrix[n_row, select] = -1. / sum(select)
            n_row = n_row + 1
    return indicator_matrix


def centering(size):
    """ generates a centering matrix
        Args:
            size (int): size of the center matrix
        Returns:
            centering_matrix (numpy.ndarray): size * size
    """
    centering_matrix = np.identity(size) - np.ones(size) / size
    return centering_matrix
