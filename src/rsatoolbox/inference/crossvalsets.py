#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generation of crossvalidation splits
"""

from copy import deepcopy
import numpy as np
from rsatoolbox.util.rdm_utils import add_pattern_index
from rsatoolbox.util.inference_util import default_k_pattern, default_k_rdm


def sets_leave_one_out_pattern(rdms, pattern_descriptor):
    """ generates training and test set combinations by leaving one level
    of pattern_descriptor out as a test set.
    This is only sensible if pattern_descriptor already defines larger groups!

    the ceil_train_set contains the rdms for the test-patterns from the
    training-rdms. This is required for computing the noise-ceiling

    Args:
        rdms(rsatoolbox.rdm.RDMs): rdms to use
        pattern_descriptor(String): descriptor to select groups

    Returns:
        train_set(list): list of tuples (rdms, pattern_idx)
        test_set(list): list of tuples (rdms, pattern_idx)
        ceil_set(list): list of tuples (rdms, pattern_idx)

    """
    pattern_descriptor, pattern_select = \
        add_pattern_index(rdms, pattern_descriptor)
    train_set = []
    test_set = []
    ceil_set = []
    for i_pattern in pattern_select:
        pattern_idx_train = np.setdiff1d(pattern_select, i_pattern)
        rdms_train = rdms.subset_pattern(pattern_descriptor,
                                         pattern_idx_train)
        pattern_idx_test = [i_pattern]
        rdms_test = rdms.subset_pattern(pattern_descriptor,
                                        pattern_idx_test)
        rdms_ceil = rdms.subset_pattern(pattern_descriptor,
                                        pattern_idx_test)
        train_set.append((rdms_train, pattern_idx_train))
        test_set.append((rdms_test, pattern_idx_test))
        ceil_set.append((rdms_ceil, pattern_idx_test))
    return train_set, test_set, ceil_set


def sets_leave_one_out_rdm(rdms, rdm_descriptor='index'):
    """ generates training and test set combinations by leaving one level
    of rdm_descriptor out as a test set.\

    Args:
        rdms(rsatoolbox.rdm.RDMs): rdms to use
        rdm_descriptor(String): descriptor to select groups

    Returns:
        train_set(list): list of tuples (rdms, pattern_idx)
        test_set(list): list of tuples (rdms, pattern_idx)
        ceil_set(list): list of tuples (rdms, pattern_idx)

    """
    rdm_select = rdms.rdm_descriptors[rdm_descriptor]
    rdm_select = np.unique(rdm_select)
    if len(rdm_select) > 1:
        train_set = []
        test_set = []
        for i_pattern in rdm_select:
            rdm_idx_train = np.setdiff1d(rdm_select, i_pattern)
            rdms_train = rdms.subset(rdm_descriptor,
                                     rdm_idx_train)
            rdm_idx_test = [i_pattern]
            rdms_test = rdms.subset(rdm_descriptor,
                                    rdm_idx_test)
            train_set.append((rdms_train, np.arange(rdms.n_cond)))
            test_set.append((rdms_test, np.arange(rdms.n_cond)))
        ceil_set = train_set
    else:
        Warning('leave one out called with only one group')
        train_set = [(rdms, np.arange(rdms.n_cond))]
        test_set = [(rdms, np.arange(rdms.n_cond))]
        ceil_set = [(rdms, np.arange(rdms.n_cond))]
    return train_set, test_set, ceil_set


def sets_k_fold(rdms, k_rdm=None, k_pattern=None, random=True,
                pattern_descriptor='index', rdm_descriptor='index'):
    """ generates training and test set combinations by splitting into k
    similar sized groups. This version splits both over rdms and over patterns
    resulting in k_rdm * k_pattern (training, test) pairs.

    If a k is set to 1 the corresponding dimension is not crossvalidated.

    Args:
        rdms(rsatoolbox.rdm.RDMs): rdms to use
        pattern_descriptor(String): descriptor to select pattern groups
        rdm_descriptor(String): descriptor to select rdm groups
        k_rdm(int): number of rdm groups
        k_pattern(int): number of pattern groups
        random(bool): whether the assignment shall be randomized

    Returns:
        train_set(list): list of tuples (rdms, pattern_idx)
        test_set(list): list of tuples (rdms, pattern_idx)
        ceil_set(list): list of tuples (rdms, pattern_idx)

    """
    rdm_select = rdms.rdm_descriptors[rdm_descriptor]
    rdm_select = np.unique(rdm_select)
    if k_rdm is None:
        k_rdm = default_k_rdm(len(rdm_select))
    pattern_descriptor, pattern_select = \
        add_pattern_index(rdms, pattern_descriptor)
    if k_pattern is None:
        k_pattern = default_k_pattern(len(pattern_select))
    assert k_rdm <= len(rdm_select), \
        'Can make at most as many groups as rdms'
    if random:
        np.random.shuffle(rdm_select)
    group_size_rdm = np.floor(len(rdm_select) / k_rdm)
    additional_rdms = len(rdm_select) % k_rdm
    train_set = []
    test_set = []
    ceil_set = []
    for i_group in range(k_rdm):
        test_idx = np.arange(i_group * group_size_rdm,
                             (i_group + 1) * group_size_rdm)
        if i_group < additional_rdms:
            test_idx = np.concatenate((test_idx, [len(rdm_select)-(i_group+1)]))
        if k_rdm <= 1:
            train_idx = test_idx
        else:
            train_idx = np.setdiff1d(np.arange(len(rdm_select)),
                                     test_idx)
        rdm_idx_test = [rdm_select[int(idx)] for idx in test_idx]
        rdm_idx_train = [rdm_select[int(idx)] for idx in train_idx]
        rdms_test = rdms.subsample(rdm_descriptor,
                                   rdm_idx_test)
        rdms_train = rdms.subsample(rdm_descriptor,
                                    rdm_idx_train)
        train_new, test_new, _ = sets_k_fold_pattern(
            rdms_train, k=k_pattern,
            pattern_descriptor=pattern_descriptor, random=random)
        ceil_new = deepcopy(test_new)
        for i_pattern in range(k_pattern):
            test_new[i_pattern][0] = rdms_test.subset_pattern(
                by=pattern_descriptor,
                value=test_new[i_pattern][1])
        train_set += train_new
        test_set += test_new
        ceil_set += ceil_new
    return train_set, test_set, ceil_set


def sets_k_fold_rdm(rdms, k_rdm=None, random=True, rdm_descriptor='index'):
    """ generates training and test set combinations by splitting into k
    similar sized groups. This version splits both over rdms and over patterns
    resulting in k_rdm * k_pattern (training, test) pairs.

    Args:
        rdms(rsatoolbox.rdm.RDMs): rdms to use
        rdm_descriptor(String): descriptor to select rdm groups
        k_rdm(int): number of rdm groups
        random(bool): whether the assignment shall be randomized

    Returns:
        train_set(list): list of tuples (rdms, pattern_idx)
        test_set(list): list of tuples (rdms, pattern_idx)

    """
    rdm_select = rdms.rdm_descriptors[rdm_descriptor]
    rdm_select = np.unique(rdm_select)
    if k_rdm is None:
        k_rdm = default_k_rdm(len(rdm_select))
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
            test_idx = np.concatenate((test_idx, [len(rdm_select)-(i_group+1)]))
        train_idx = np.setdiff1d(np.arange(len(rdm_select)),
                                 test_idx)
        rdm_idx_test = [rdm_select[int(idx)] for idx in test_idx]
        rdm_idx_train = [rdm_select[int(idx)] for idx in train_idx]
        rdms_test = rdms.subsample(rdm_descriptor,
                                   rdm_idx_test)
        rdms_train = rdms.subsample(rdm_descriptor,
                                    rdm_idx_train)
        train_set.append([rdms_train, np.arange(rdms_train.n_cond)])
        test_set.append([rdms_test, np.arange(rdms_train.n_cond)])
    ceil_set = train_set
    return train_set, test_set, ceil_set


def sets_k_fold_pattern(rdms, pattern_descriptor='index',
                        k=None, random=False):
    """ generates training and test set combinations by splitting into k
    similar sized groups. This version splits in the given order or
    randomizes the order. For k=1 training and test_set are whole dataset,
    i.e. no crossvalidation is performed.

    For only crossvalidating over patterns there is no independent training
    set for calculating a noise ceiling for the patterns.
    To express this we set ceil_set to None, which makes the crossvalidation
    function calculate a leave one rdm out noise ceiling for the right
    patterns instead.

    Args:
        rdms(rsatoolbox.rdm.RDMs): rdms to use
        pattern_descriptor(String): descriptor to select groups
        k(int): number of groups
        random(bool): whether the assignment shall be randomized

    Returns:
        train_set(list): list of tuples (rdms, pattern_idx)
        test_set(list): list of tuples (rdms, pattern_idx)
        ceil_set = None

    """
    pattern_descriptor, pattern_select = \
        add_pattern_index(rdms, pattern_descriptor)
    if k is None:
        k = default_k_pattern(len(pattern_select))
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
            test_idx = np.concatenate((test_idx, [len(pattern_select)-(i_group+1)]))
        if k <= 1:
            train_idx = test_idx
        else:
            train_idx = np.setdiff1d(np.arange(len(pattern_select)),
                                     test_idx)
        pattern_idx_test = [pattern_select[int(idx)] for idx in test_idx]
        pattern_idx_train = [pattern_select[int(idx)] for idx in train_idx]
        rdms_test = rdms.subset_pattern(pattern_descriptor,
                                        pattern_idx_test)
        rdms_train = rdms.subset_pattern(pattern_descriptor,
                                         pattern_idx_train)
        test_set.append([rdms_test, pattern_idx_test])
        train_set.append([rdms_train, pattern_idx_train])
    ceil_set = None
    return train_set, test_set, ceil_set


def sets_of_k_rdm(rdms, rdm_descriptor='index', k=5, random=False):
    """ generates training and test set combinations by splitting into
    groups of k. This version splits in the given order or
    randomizes the order. If the number of patterns is not divisible by k
    patterns are added to the first groups such that those have k+1 patterns

    Args:
        rdms(rsatoolbox.rdm.RDMs): rdms to use
        pattern_descriptor(String): descriptor to select groups
        k(int): number of groups
        random(bool): whether the assignment shall be randomized

    Returns:
        train_set(list): list of tuples (rdms, pattern_idx)
        test_set(list): list of tuples (rdms, pattern_idx)
        ceil_set(list): list of tuples (rdms, pattern_idx)

    """
    rdm_select = rdms.rdm_descriptors[rdm_descriptor]
    rdm_select = np.unique(rdm_select)
    assert k <= len(rdm_select) / 2, \
        'to form groups we can use at most half the patterns per group'
    n_groups = int(len(rdm_select) / k)
    return sets_k_fold_rdm(rdms, rdm_descriptor=rdm_descriptor,
                           k=n_groups, random=random)


def sets_of_k_pattern(rdms, pattern_descriptor=None, k=5, random=False):
    """ generates training and test set combinations by splitting into
    groups of k. This version splits in the given order or
    randomizes the order. If the number of patterns is not divisible by k
    patterns are added to the first groups such that those have k+1 patterns

    Args:
        rdms(rsatoolbox.rdm.RDMs): rdms to use
        pattern_descriptor(String): descriptor to select groups
        k(int): number of groups
        random(bool): whether the assignment shall be randomized

    Returns:
        train_set(list): list of tuples (rdms, pattern_idx)
        test_set(list): list of tuples (rdms, pattern_idx)

    """
    pattern_descriptor, pattern_select = \
        add_pattern_index(rdms, pattern_descriptor)
    assert k <= len(pattern_select) / 2, \
        'to form groups we can use at most half the patterns per group'
    n_groups = int(len(pattern_select) / k)
    return sets_k_fold_pattern(rdms, pattern_descriptor=pattern_descriptor,
                               k=n_groups, random=random)


def sets_random(rdms, n_rdm=None, n_pattern=None, n_cv=2,
                pattern_descriptor='index', rdm_descriptor='index'):
    """ generates training and test set combinations by selecting random
    test sets of n_rdm RDMs and n_pattern patterns and using the rest of
    the data as the training set.

    If a n is set to 0 the corresponding dimension is not crossvalidated.

    Args:
        rdms(rsatoolbox.rdm.RDMs): rdms to split
        pattern_descriptor(String): descriptor to select pattern groups
        rdm_descriptor(String): descriptor to select rdm groups
        n_rdm(int): number of rdms per test set
        n_pattern(int): number of patterns per test set

    Returns:
        train_set(list): list of tuples (rdms, pattern_idx)
        test_set(list): list of tuples (rdms, pattern_idx)
        ceil_set(list): list of tuples (rdms, pattern_idx)

    """
    rdm_select = rdms.rdm_descriptors[rdm_descriptor]
    rdm_select = np.unique(rdm_select)
    if n_rdm is None:
        k_rdm = default_k_rdm(len(rdm_select))
        n_rdm = int(np.floor(len(rdm_select) / k_rdm))
    pattern_descriptor, pattern_select = \
        add_pattern_index(rdms, pattern_descriptor)
    if n_pattern is None:
        k_pattern = default_k_pattern(len(pattern_select))
        n_pattern = int(np.floor(len(pattern_select) / k_pattern))
    train_set = []
    test_set = []
    ceil_set = []
    for _i_group in range(n_cv):
        # shuffle
        np.random.shuffle(rdm_select)
        np.random.shuffle(pattern_select)
        # choose indices based on n_rdm
        if n_rdm == 0:
            train_idx = np.arange(len(rdm_select))
            test_idx = np.arange(len(rdm_select))
        else:
            test_idx = np.arange(n_rdm)
            train_idx = np.arange(n_rdm, len(rdm_select))
        # take subset of rdms
        rdm_idx_test = [rdm_select[int(idx)] for idx in test_idx]
        rdm_idx_train = [rdm_select[int(idx)] for idx in train_idx]
        rdms_test = rdms.subsample(rdm_descriptor,
                                   rdm_idx_test)
        rdms_train = rdms.subsample(rdm_descriptor,
                                    rdm_idx_train)
        # choose indices based on n_pattern
        if n_pattern == 0:
            train_idx = np.arange(len(pattern_select))
            test_idx = np.arange(len(pattern_select))
        else:
            test_idx = np.arange(n_pattern)
            train_idx = np.arange(n_pattern, len(pattern_select))
        pattern_idx_test = [pattern_select[int(idx)] for idx in test_idx]
        pattern_idx_train = [pattern_select[int(idx)] for idx in train_idx]
        rdms_test = rdms_test.subset_pattern(pattern_descriptor,
                                             pattern_idx_test)
        rdms_ceil = rdms_train.subset_pattern(pattern_descriptor,
                                              pattern_idx_test)
        rdms_train = rdms_train.subset_pattern(pattern_descriptor,
                                               pattern_idx_train)
        test_set.append([rdms_test, pattern_idx_test])
        train_set.append([rdms_train, pattern_idx_train])
        ceil_set.append([rdms_ceil, pattern_idx_test])
    return train_set, test_set, ceil_set
