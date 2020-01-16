#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
inference module: evaluate models
@author: heiko
"""

import numpy as np
from pyrsa.rdm import compare
from pyrsa.inference import bootstrap_sample_rdm


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
    rdm_pred = model.predict(theta=theta)
    return compare(rdm_pred, data, method)


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
        evaluations[i] = eval_fixed(model, sample, theta=theta,
                                    method=method)
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

        pattern_descriptor(string): rdm_descriptor to group rdms for bootstrap

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
        evaluations.append(compare(pred, test[0], method))
    return evaluations


def sets_leave_one_out(rdms, pattern_descriptor=None):
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
        train_set.append((rdms_train, pattern_sample_train))
        test_set.append((rdms_test, pattern_sample_test))
    return train_set, test_set
