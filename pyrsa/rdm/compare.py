#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 13:01:12 2019

@author: heiko
"""
import numpy as np


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
    vector1 = rdm1.get_vectors()
    vector2 = rdm2.get_vectors()
    if not (vector1.shape[1] == vector2.shape[1]):
        raise ValueError('rdm1 and rdm2 must be RDMs of equal shape')
    vector1 = np.mean(vector1, 0)
    vector2 = np.mean(vector2, 0)
    dist = 1 - (np.sum(vector1*vector2) /
                np.sqrt(np.sum(vector1 * vector1)) /
                np.sqrt(np.sum(vector2 * vector2)))
    return dist
