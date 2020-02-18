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
from pyrsa.util.rdm_utils import add_pattern_index
from .crossvalsets import sets_k_fold


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
        pattern_descriptor, pattern_select = \
            add_pattern_index(rdm_pred, pattern_descriptor)
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
        pattern_descriptor, pattern_select = \
            add_pattern_index(rdm_pred, pattern_descriptor)
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
        train_set(list): a list of the training RDMs with 2-tuple entries:
            (RDMs, pattern_sample)
        test_set(list): a list of the test RDMs with 2-tuple entries:
            (RDMs, pattern_sample)
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


def _concat_sampling(sample1, sample2):
    """ computes an index vector for the sequential sampling with sample1 
    and sample2
    """
    sample_out = [[i_samp1 for i_samp1 in sample1 if i_samp1==i_samp2]
                  for i_samp2 in sample2]
    return sum(sample_out,[])