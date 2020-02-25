#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 09:44:04 2020

@author: heiko
"""

import numpy as np
from pyrsa.util.rdm_utils import add_pattern_index


def sets_leave_one_out_pattern(rdms, pattern_descriptor=None):
    """ generates training and test set combinations by leaving one level
    of pattern_descriptor out as a test set.
    This is only sensible if pattern_descriptor already defines larger groups!

    Args:
        rdms(pyrsa.rdm.RDMs): rdms to use
        pattern_descriptor(String): descriptor to select groups

    Returns:
        train_set(list): list of tuples (rdms, pattern_sample)
        test_set(list): list of tuples (rdms, pattern_sample)

    """
    pattern_descriptor, pattern_select = \
        add_pattern_index(rdms, pattern_descriptor)
    train_set = []
    test_set = []
    for i_pattern in pattern_select:
        pattern_sample_train = np.setdiff1d(pattern_select, i_pattern)
        rdms_train = rdms.subset_pattern(pattern_descriptor,
                                         pattern_sample_train)
        pattern_sample_test = [i_pattern]
        rdms_test = rdms.subset_pattern(pattern_descriptor,
                                        pattern_sample_test)
        train_set.append((rdms_train, pattern_sample_train))
        test_set.append((rdms_test, pattern_sample_test))
    return train_set, test_set


def sets_k_fold(rdms, k_rdm=5, k_pattern=5, random=True,
                pattern_descriptor=None, rdm_descriptor=None):
    """ generates training and test set combinations by splitting into k
    similar sized groups. This version splits both over rdms and over patterns
    resulting in k_rdm * k_pattern (training, test) pairs.

    Args:
        rdms(pyrsa.rdm.RDMs): rdms to use
        pattern_descriptor(String): descriptor to select pattern groups
        rdm_descriptor(String): descriptor to select rdm groups
        k_rdm(int): number of rdm groups
        k_pattern(int): number of pattern groups
        random(bool): whether the assignment shall be randomized

    Returns:
        train_set(list): list of tuples (rdms, pattern_sample)
        test_set(list): list of tuples (rdms, pattern_sample)

    """
    if rdm_descriptor is None:
        rdm_select = np.arange(rdms.n_rdm)
        rdms.rdm_descriptors['index'] = rdm_select
        pattern_descriptor = 'index'
    else:
        rdm_select = rdms.rdm_descriptors[rdm_descriptor]
        rdm_select = np.unique(rdm_select)
    assert k_rdm <= len(rdm_select), \
        'Can make at most as many groups as rdms'
    if random:
        np.random.shuffle(rdm_select)
    group_size_rdm = np.floor(len(rdm_select) / k_rdm)
    additional_rdms = len(rdm_select) % k_rdm
    train_set = []
    test_set = []
    for i_group in range(k_rdm):
        test_idx = np.arange(i_group * group_size_rdm,
                             (i_group + 1) * group_size_rdm)
        if i_group < additional_rdms:
            test_idx = np.concatenate((test_idx, [-(i_group+1)]))
        train_idx = np.setdiff1d(np.arange(len(rdm_select)),
                                 test_idx)
        rdm_sample_test = [rdm_select[int(idx)] for idx in test_idx]
        rdm_sample_train = [rdm_select[int(idx)] for idx in train_idx]
        rdms_test = rdms.subsample(rdm_descriptor,
                                   rdm_sample_test)
        rdms_train = rdms.subsample(rdm_descriptor,
                                    rdm_sample_train)
        train_new, test_new = sets_k_fold_pattern(rdms_train, k=k_pattern,
            pattern_descriptor=pattern_descriptor, random=random)
        for i_pattern in range(k_pattern):
            test_new[i_pattern][0] = rdms_test.subsample_pattern(
                by=pattern_descriptor,
                value=test_new[i_pattern][1])
        train_set += train_new
        test_set += test_new
    return train_set, test_set


def sets_k_fold_rdm(rdms, k_rdm=5, random=True, rdm_descriptor=None):
    """ generates training and test set combinations by splitting into k
    similar sized groups. This version splits both over rdms and over patterns
    resulting in k_rdm * k_pattern (training, test) pairs.

    Args:
        rdms(pyrsa.rdm.RDMs): rdms to use
        rdm_descriptor(String): descriptor to select rdm groups
        k_rdm(int): number of rdm groups
        random(bool): whether the assignment shall be randomized

    Returns:
        train_set(list): list of tuples (rdms, pattern_sample)
        test_set(list): list of tuples (rdms, pattern_sample)

    """
    if rdm_descriptor is None:
        rdm_select = np.arange(rdms.n_rdm)
        rdms.rdm_descriptors['index'] = rdm_select
    else:
        rdm_select = rdms.rdm_descriptors[rdm_descriptor]
        rdm_select = np.unique(rdm_select)
    assert k_rdm <= len(rdm_select), \
        'Can make at most as many groups as rdms'
    if random:
        np.random.shuffle(rdm_select)
    group_size_rdm = np.floor(len(rdm_select) / k_rdm)
    additional_rdms = len(rdm_select) % k_rdm
    train_set = []
    test_set = []
    for i_group in range(k_rdm):
        test_idx = np.arange(i_group * group_size_rdm,
                             (i_group + 1) * group_size_rdm)
        if i_group < additional_rdms:
            test_idx = np.concatenate((test_idx, [-(i_group+1)]))
        train_idx = np.setdiff1d(np.arange(len(rdm_select)),
                                 test_idx)
        rdm_sample_test = [rdm_select[int(idx)] for idx in test_idx]
        rdm_sample_train = [rdm_select[int(idx)] for idx in train_idx]
        rdms_test = rdms.subsample(rdm_descriptor,
                                   rdm_sample_test)
        rdms_train = rdms.subsample(rdm_descriptor,
                                    rdm_sample_train)
        train_set.append([rdms_train, np.arange(rdms_train.n_cond)])
        test_set.append([rdms_test, np.arange(rdms_train.n_cond)])
    return train_set, test_set
    


def sets_k_fold_pattern(rdms, pattern_descriptor=None, k=5, random=False):
    """ generates training and test set combinations by splitting into k
    similar sized groups. This version splits in the given order or 
    randomizes the order

    Args:
        rdms(pyrsa.rdm.RDMs): rdms to use
        pattern_descriptor(String): descriptor to select groups
        k(int): number of groups
        random(bool): whether the assignment shall be randomized

    Returns:
        train_set(list): list of tuples (rdms, pattern_sample)
        test_set(list): list of tuples (rdms, pattern_sample)

    """
    pattern_descriptor, pattern_select = \
        add_pattern_index(rdms, pattern_descriptor)
    assert k <= len(pattern_select), \
        'Can make at most as many groups as conditions'
    if random:
        np.random.shuffle(pattern_select)
    group_size = np.floor(len(pattern_select) / k)
    additional_patterns = len(pattern_select) % k
    train_set = []
    test_set = []
    for i_group in range(k):
        test_idx = np.arange(i_group * group_size,
                             (i_group + 1) * group_size)
        if i_group < additional_patterns:
            test_idx = np.concatenate((test_idx, [-(i_group+1)]))
        train_idx = np.setdiff1d(np.arange(len(pattern_select)),
                                 test_idx)
        pattern_sample_test = [pattern_select[int(idx)] for idx in test_idx]
        pattern_sample_train = [pattern_select[int(idx)] for idx in train_idx]
        rdms_test = rdms.subset_pattern(pattern_descriptor,
                                        pattern_sample_test)
        rdms_train = rdms.subset_pattern(pattern_descriptor,
                                         pattern_sample_train)
        test_set.append([rdms_test, pattern_sample_test])
        train_set.append([rdms_train, pattern_sample_train])
    return train_set, test_set


def sets_of_k_pattern(rdms, pattern_descriptor=None, k=5, random=False):
    """ generates training and test set combinations by splitting into
    groups of k. This version splits in the given order or 
    randomizes the order. If the number of patterns is not divisible by k
    patterns are added to the first groups such that those have k+1 patterns

    Args:
        rdms(pyrsa.rdm.RDMs): rdms to use
        pattern_descriptor(String): descriptor to select groups
        k(int): number of groups
        random(bool): whether the assignment shall be randomized

    Returns:
        train_set(list): list of tuples (rdms, pattern_sample)
        test_set(list): list of tuples (rdms, pattern_sample)

    """
    pattern_descriptor, pattern_select = \
        add_pattern_index(rdms, pattern_descriptor)
    assert k <= len(pattern_select) / 2, \
        'to form groups we can use at most half the patterns per group'
    n_groups = int(len(pattern_select) / k)
    return sets_k_fold_pattern(rdms, pattern_descriptor=pattern_descriptor,
                               k=n_groups, random=random)
