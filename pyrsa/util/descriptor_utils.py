#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Descriptor handling

@author: heiko
"""

import numpy as np


def bool_index(descriptor, value):
    """
    creates a boolean index vector where a descriptor has a value

        Args:
            descriptor(np.ndarray): descriptor vector
            value:                  value or list of values to mark

        Returns:
            bool_index:         boolean index vector where descriptor == value
    """
    if type(value) is list or type(value) is tuple:
        bool_index = np.array([descriptor == v for v in value])
        bool_index = np.any(bool_index, axis=0)
    else:
        bool_index = np.array(descriptor == value)
    return bool_index
