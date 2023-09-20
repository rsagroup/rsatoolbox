#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Collection of helper methods for data module

@author: baihan
"""

from collections.abc import Iterable
import numpy as np


def extract_dict(dictionary, indices):
    """extract key-value pairs with values given indexes.
    """
    extracted_dictionary = dictionary.copy()
    for k, v in dictionary.items():
        if isinstance(indices, Iterable):
            extracted_dictionary[k] = [v[idx] for idx in indices]
        else:
            extracted_dictionary[k] = v[indices]
    return extracted_dictionary


def get_unique_unsorted(array):
    """return a unique unsorted list
    """
    u, indices = np.unique(array, return_index=True)
    temp = indices.argsort()
    return u[temp]


def get_unique_inverse(array):
    """return a unique list in original order + inverse index to get
    which entries correspond to which unique value
    """
    u, indices, inverse = np.unique(array, return_index=True, return_inverse=True)
    # sort indices to remove sorting of np.unique
    temp = indices.argsort()
    # invert sorting permutation
    s = np.empty(temp.size, temp.dtype)
    s[temp] = np.arange(temp.size)
    return u[temp], s[inverse]
