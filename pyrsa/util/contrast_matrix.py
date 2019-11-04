#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Contrast matrix
@author: heiko
"""

import numpy as np

def contrast_matrix(n_patterns):
    """
    generates a matrix C which maps from measurement space into 
    the space of differences.
    
        Args:
            n_pattern: Number of patterns
            
        Returns:
            C: Contrast matrix
    """
    C = np.zeros((int((n_patterns*(n_patterns-1)/2)),n_patterns))
    k=0
    for i in range(n_patterns):
        for j in range(i+1,n_patterns):
            C[k,i] = 1
            C[k,j] = -1
            k=k+1
    return C