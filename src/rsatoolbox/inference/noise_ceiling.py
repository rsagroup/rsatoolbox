#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
calculation of noise ceilings
"""

import numpy as np
from rsatoolbox.util.inference_util import pool_rdm
from rsatoolbox.rdm import compare
from .crossvalsets import sets_leave_one_out_rdm


def cv_noise_ceiling(rdms, ceil_set, test_set, method='cosine',
                     pattern_descriptor='index'):
    """ calculates the noise ceiling for crossvalidation.
    The upper bound is calculated by pooling all rdms for the appropriate
    patterns in the testsets.
    the lower bound is calculated by using only the appropriate rdms
    from ceil_set for training.

    Args:
        rdms(rsatoolbox.rdm.RDMs): complete data
        ceil_set(list): a list of the training RDMs with 2-tuple entries:
            (RDMs, pattern_idx)
        test_set(list): a list of the test RDMs with 2-tuple entries:
            (RDMs, pattern_idx)
        method(string): comparison method to use
        pattern_descriptor(string): descriptor to group patterns

    Returns:
        list: lower nc-bound, upper nc-bound

    """
    assert len(ceil_set) == len(test_set), \
        'train_set and test_set must have the same length'
    noise_min = []
    noise_max = []
    for i in range(len(ceil_set)):
        train = ceil_set[i]
        test = test_set[i]
        pred_train = pool_rdm(train[0], method=method)
        pred_train = pred_train.subsample_pattern(by=pattern_descriptor,
                                                  value=test[1])
        pred_test = pool_rdm(rdms, method=method)
        pred_test = pred_test.subsample_pattern(by=pattern_descriptor,
                                                value=test[1])
        noise_min.append(np.mean(compare(pred_train, test[0], method)))
        noise_max.append(np.mean(compare(pred_test, test[0], method)))
    noise_min = np.mean(np.array(noise_min))
    noise_max = np.mean(np.array(noise_max))
    return noise_min, noise_max


def boot_noise_ceiling(rdms, method='cosine', rdm_descriptor='index'):
    """ calculates a noise ceiling by leave one out & full set

    Args:
        rdms(rsatoolbox.rdm.RDMs): data to calculate noise ceiling
        method(string): comparison method to use
        rdm_descriptor(string): descriptor to group rdms

    Returns:
        list: [lower nc-bound, upper nc-bound]

    """
    _, test_set, ceil_set = sets_leave_one_out_rdm(rdms, rdm_descriptor)
    pred_test = pool_rdm(rdms, method=method)
    noise_min = []
    noise_max = []
    for i in range(len(ceil_set)):
        train = ceil_set[i]
        test = test_set[i]
        pred_train = pool_rdm(train[0], method=method)
        noise_min.append(np.mean(compare(pred_train, test[0], method)))
        noise_max.append(np.mean(compare(pred_test, test[0], method)))
    noise_min = np.mean(np.array(noise_min))
    noise_max = np.mean(np.array(noise_max))
    return noise_min, noise_max
