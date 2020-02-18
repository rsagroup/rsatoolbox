#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 10:05:47 2020

@author: heiko
"""

import numpy as np
from pyrsa.util.inference_util import pool_rdm
from pyrsa.rdm import compare


def noise_ceiling(train_set, test_set, method='cosine',
                  pattern_descriptor=None):
    """ calculates the noise ceiling for a train_set, test_set pair defined 
    as for crossvalidation

    Args:
        train_set(list): a list of the training RDMs with 2-tuple entries:
            (RDMs, pattern_sample)
        test_set(list): a list of the test RDMs with 2-tuple entries:
            (RDMs, pattern_sample)
        method(string): comparison method to use
        pattern_descriptor(string): descriptor to group patterns

    Returns:
        numpy.ndarray: [lower nc-bound, upper nc-bound]

    """
    assert len(train_set) == len(test_set), \
        'train_set and test_set must have the same length'
    if pattern_descriptor is None:
        pattern_descriptor = 'index'
    noise_min = []
    noise_max = []
    for i in range(len(train_set)):
        train = train_set[i]
        test = test_set[i]
        pred_train = pool_rdm(train[0], method=method)
        pred_train = pred_train.subsample_pattern(by=pattern_descriptor, 
                                                  value=test[1])
        pred_test = pool_rdm(test[0], method=method)
        pred_test = pred_test.subsample_pattern(by=pattern_descriptor, 
                                                value=test[1])
        noise_min.append(np.mean(compare(pred_train, test[0], method)))
        noise_max.append(np.mean(compare(pred_test, test[0], method)))
    noise_min = np.mean(np.array(noise_min))
    noise_max = np.mean(np.array(noise_max))
    return [noise_min,noise_max]
