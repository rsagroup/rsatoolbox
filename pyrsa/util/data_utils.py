#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Collection of helper methods for data module
    get_unique_unsorted: return a unique unsorted list
@author: baihan
"""

import numpy as np
import pyrsa as rsa


def get_unique_unsorted(array):
    u, indices = np.unique(array, return_index=True)
    temp = indices.argsort()
    ranks = np.empty_like(temp)
    ranks[temp] = np.arange(len(indices))
    return u[ranks]
