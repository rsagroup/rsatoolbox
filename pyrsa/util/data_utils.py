#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Collection of helper methods for data module
"""

import numpy as np


def extract_dict(dictionary, indices):
    """extract key-value pairs with values given indexes.
    """
    extracted_dictionary = dictionary.copy()
    for k, v in dictionary.items():
        extracted_dictionary[k] = np.array(v)[indices]
    return extracted_dictionary


def get_unique_unsorted(array):
    """return a unique unsorted list
    """
    u, indices = np.unique(array, return_index=True)
    temp = indices.argsort()
    return u[temp]
