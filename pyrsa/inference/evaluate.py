#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
inference module: evaluate models
@author: heiko
"""

import numpy as np
from pyrsa.rdm import compare
from pyrsa.inference import bootstrap_sample
from pyrsa.inference import bootstrap_sample_rdm
from pyrsa.inference import bootstrap_sample_pattern


def eval_fixed(model, data, theta=None, method='cosine'):
    """evaluates a model on data, without any bootstrapping or
    cross-validation

    Args:
        model(pyrsa.model.Model): Model to be evaluated
        data(pyrsa.rdm.RDMs): data to evaluate on
        theta(numpy.ndarray): parameter vector for the model
        method(string): comparison method to use

    Returns:
        float: evaluation

    """
    rdm_pred = model.predict_rdm(theta=theta)
    return compare(rdm_pred, data, method)


def eval_bootstrap(model, data, theta=None, method='cosine', N=1000,
                   pattern_descriptor=None, rdm_descriptor=None):
    """evaluates a model on data
    performs bootstrapping to get a sampling distribution

    Args:
        model(pyrsa.model.Model): Model to be evaluated
        data(pyrsa.rdm.RDMs): data to evaluate on
        theta(numpy.ndarray): parameter vector for the model
        method(string): comparison method to use
        N(int): number of samples
        pattern_descriptor(string): descriptor to group patterns for bootstrap
        rdm_descriptor(string): descriptor to group rdms for bootstrap

    Returns:
        numpy.ndarray: vector of evaluations

    """
    evaluations = np.zeros(N)
    for i in range(N):
        sample, rdm_sample, pattern_sample = \
            bootstrap_sample(data, rdm_descriptor=rdm_descriptor,
                             pattern_descriptor=pattern_descriptor)
        rdm_pred = model.predict_rdm(theta=theta)
        if pattern_descriptor is None:
            rdm_pred.pattern_descriptors['index'] = np.arange(rdm_pred.n_cond)
            rdm_pred = rdm_pred.subsample_pattern('index',
                                                  pattern_sample)
        else:
            rdm_pred = rdm_pred.subsample_pattern(pattern_descriptor,
                                                  pattern_sample)
        evaluations[i] = np.mean(compare(rdm_pred, sample, method))
    return evaluations


def eval_bootstrap_pattern(model, data, theta=None, method='cosine', N=1000,
                           pattern_descriptor=None):
    """evaluates a model on data
    performs bootstrapping to get a sampling distribution

    Args:
        model(pyrsa.model.Model): Model to be evaluated
        data(pyrsa.rdm.RDMs): data to evaluate on
        theta(numpy.ndarray): parameter vector for the model
        method(string): comparison method to use
        N(int): number of samples
        pattern_descriptor(string): descriptor to group patterns for bootstrap

    Returns:
        numpy.ndarray: vector of evaluations

    """
    evaluations = np.zeros(N)
    for i in range(N):
        sample, pattern_sample = \
            bootstrap_sample_pattern(data, pattern_descriptor)
        rdm_pred = model.predict_rdm(theta=theta)
        if pattern_descriptor is None:
            rdm_pred.pattern_descriptors['index'] = np.arange(rdm_pred.n_cond)
            rdm_pred = rdm_pred.subsample_pattern('index',
                                                  pattern_sample)
        else:
            rdm_pred = rdm_pred.subsample_pattern(pattern_descriptor,
                                                  pattern_sample)
        evaluations[i] = np.mean(compare(rdm_pred, sample, method))
    return evaluations


def eval_bootstrap_rdm(model, data, theta=None, method='cosine', N=1000,
                       rdm_descriptor=None):
    """evaluates a model on data
    performs bootstrapping to get a sampling distribution

    Args:
        model(pyrsa.model.Model): Model to be evaluated
        data(pyrsa.rdm.RDMs): data to evaluate on
        theta(numpy.ndarray): parameter vector for the model
        method(string): comparison method to use
        N(int): number of samples
        rdm_descriptor(string): rdm_descriptor to group rdms for bootstrap

    Returns:
        numpy.ndarray: vector of evaluations

    """
    evaluations = np.zeros(N)
    for i in range(N):
        sample = bootstrap_sample_rdm(data, rdm_descriptor)
        evaluations[i] = np.mean(eval_fixed(model, sample[0], theta=theta,
                                            method=method))
    return evaluations


def crossval(model, train_set, test_set, method='cosine', fitter=None,
             pattern_descriptor=None):
    """evaluates a model on cross-validation sets

    Args:
        model(pyrsa.model.Model): Model to be evaluated
        train_set(list): a list of the training RDMs with 3-tuple entries:
            (RDMs, pattern_sample, pattern_select)
        test_set(list): a list of the test RDMs with 3-tuple entries:
            (RDMs, pattern_sample, pattern_select)
        method(string): comparison method to use
        pattern_descriptor(string): descriptor to group patterns

    Returns:
        numpy.ndarray: vector of evaluations

    """
    assert len(train_set) == len(test_set), \
        'train_set and test_set must have the same length'
    if fitter is None:
        fitter = model.default_fitter
    if pattern_descriptor is None:
        pattern_descriptor = 'index'
    evaluations = []
    for i in range(len(train_set)):
        train = train_set[i]
        test = test_set[i]
        theta = fitter(model, train[0], method=method,
                       pattern_sample=train[1],
                       pattern_descriptor=pattern_descriptor)
        pred = model.predict_rdm(theta)
        pred = pred.subsample_pattern(by=pattern_descriptor, value=test[1])
        evaluations.append(np.mean(compare(pred, test[0], method)))
    return np.array(evaluations)


def bootstrap_crossval(model, data, method='cosine', fitter=None,
                       k_pattern=5, k_rdm=5, N=1000,
                       pattern_descriptor=None, rdm_descriptor=None,
                       random=True):
    """evaluates a model by k-fold crossvalidation within a bootstrap

    Args:
        model(pyrsa.model.Model): Model to be evaluated
        datat(pyrsa.rdm.RDMs): RDM data to use
        train_set(list): a list of the training RDMs with 3-tuple entries:
            (RDMs, pattern_sample, pattern_select)
        test_set(list): a list of the test RDMs with 3-tuple entries:
            (RDMs, pattern_sample, pattern_select)
        method(string): comparison method to use
        pattern_descriptor(string): descriptor to group patterns
        rdm_descriptor(string): descriptor to group rdms

    Returns:
        numpy.ndarray: matrix of evaluations (N x k)
    """
    evaluations = np.zeros((N, k_pattern*k_rdm))
    for i_sample in range(N):
        sample, rdm_sample, pattern_sample = bootstrap_sample(data,
            rdm_descriptor=rdm_descriptor,
            pattern_descriptor=pattern_descriptor)
        train_set, test_set = sets_k_fold(sample,
            pattern_descriptor=pattern_descriptor,
            rdm_descriptor=rdm_descriptor,
            k_pattern=k_pattern, k_rdm=k_rdm, random=random)
        for idx in range(len(test_set)):
            test_set[idx][1] = _concat_sampling(pattern_sample,
                                                test_set[idx][1])
            train_set[idx][1] = _concat_sampling(pattern_sample,
                                                 train_set[idx][1])
        evaluations[i_sample,:] = crossval(model, train_set, test_set,
            method=method,
            fitter=fitter, 
            pattern_descriptor=pattern_descriptor)
    return evaluations


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
    if pattern_descriptor is None:
        pattern_select = np.arange(rdms.n_cond)
        rdms.pattern_descriptors['index'] = pattern_select
        pattern_descriptor = 'index'
    else:
        pattern_select = rdms.pattern_descriptors[pattern_descriptor]
        pattern_select = np.unique(pattern_select)
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
    if pattern_descriptor is None:
        pattern_select = np.arange(rdms.n_cond)
        rdms.pattern_descriptors['index'] = pattern_select
        pattern_descriptor = 'index'
    else:
        pattern_select = rdms.pattern_descriptors[pattern_descriptor]
        pattern_select = np.unique(pattern_select)
    assert k <= len(pattern_select) / 2, \
        'to form groups we can use at most half the patterns per group'
    n_groups = int(len(pattern_select) / k)
    return sets_k_fold_pattern(rdms, pattern_descriptor=pattern_descriptor,
                               k=n_groups, random=random)


def _concat_sampling(sample1, sample2):
    """ computes an index vector for the sequential sampling with sample1 
    and sample2
    """
    sample_out = [[i_samp1 for i_samp1 in sample1 if i_samp1==i_samp2]
                  for i_samp2 in sample2]
    return sum(sample_out,[])