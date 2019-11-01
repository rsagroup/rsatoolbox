#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Collection of helper methods for data module
    check_dict_length:   check if each value in dict matches
                         length n
    extract_dict:        extract key-value pairs with values
                         given indexes.
    get_unique_unsorted: return a unique unsorted list
@author: baihan
"""

import numpy as np
import pyrsa as rsa


def check_dict_length(dictionary, n):
    for k, v in dictionary.items():
        if v.shape[0] != n:
            return False
    return True


def extract_dict(dictionary, indices):
    extracted_dictionary = dictionary.copy()
    for k, v in dictionary.items():
        extracted_dictionary[k] = v[indices]
    return extracted_dictionary


def get_unique_unsorted(array):
    u, indices = np.unique(array, return_index=True)
    temp = indices.argsort()
    ranks = np.empty_like(temp)
    ranks[temp] = np.arange(len(indices))
    return u[ranks]


def check_descriptors_dimension(des, name, n):
    if des is not None:
        if not check_dict_length(des, n):
            raise AttributeError(
                name + " have mismatched dimension with measurements.")