#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dataset computations
@author: heiko
"""

import numpy as np

def average_dataset(dataset):
    """
    computes the average of a dataset
    
        Args:
            dataset(pyrsa.data.Dataset): the dataset to operate on 
            
        Returns:
            average(numpy.ndarray): average activation vector
    """
    return np.mean(dataset.measurements, axis = 0)
