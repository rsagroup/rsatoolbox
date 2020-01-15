#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Crossvalidation
@author: heiko
"""

import numpy as np


def leave_one_out(rdms, pattern_descriptor=None):
    """ generates training and test set combinations by leaving one level
    of pattern_descriptor out as a test set
    
    Args:
        rdms(pyrsa.rdm.RDMs): rdms to use
        
        pattern_descriptor(String): descriptor to select groups
        
    Returns:
        train_set(list): list of tuples (rdms, pattern_sample, pattern_select)
        test_set(list): list of tuples (rdms, pattern_sample, pattern_select)
    """
    if pattern_descriptor is None:
        pattern_select = np.arange(rdms.n_cond)
        rdms.pattern_descriptors['index'] = pattern_select
        pattern_descriptor = 'index'
    else:
        pattern_select = rdms.pattern_descriptors[pattern_descriptor]
        pattern_select = np.unique(pattern_select)
    train_set = []
    test_set = []
    for i_pattern in pattern_select:
        pattern_sample_train = np.setdiff1d(pattern_select, i_pattern)
        rdms_train = rdms.subset_pattern(pattern_descriptor,
                                         pattern_sample_train)
        pattern_sample_test = [i_pattern]
        rdms_test = rdms.subset_pattern(pattern_descriptor, i_pattern)
        train_set.append((rdms_train, pattern_sample_train, pattern_select))
        test_set.append((rdms_test, pattern_sample_test, pattern_select))
    return train_set, test_set
