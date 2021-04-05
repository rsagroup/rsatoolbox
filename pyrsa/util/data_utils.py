#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Collection of helper methods for data module

@author: baihan
"""

import numpy as np
from collections.abc import Iterable


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
