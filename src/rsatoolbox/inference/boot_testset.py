#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
boostrap-testset evaluation methods
variants of taking a bootstrap sample and taking the unsampled patterns as
a testset
"""

import numpy as np
from rsatoolbox.util.inference_util import input_check_model
from .bootstrap import bootstrap_sample
from .bootstrap import bootstrap_sample_rdm
from .bootstrap import bootstrap_sample_pattern
from .evaluate import crossval


def bootstrap_testset(models, data, method='cosine', fitter=None, N=1000,
                      pattern_descriptor=None, rdm_descriptor=None):
    """takes a bootstrap sample and evaluates on the rdms and patterns not
    sampled
    also returns the size of each test_set to allow later weighting
    or selection if this is desired.

    Args:
        models(rsatoolbox.model.Model): Models to be evaluated
        data(rsatoolbox.rdm.RDMs): RDM data to use
        method(string): comparison method to use
        fitter(function): fitting function
        pattern_descriptor(string): descriptor to group patterns
        rdm_descriptor(string): descriptor to group rdms

    Returns:
        numpy.ndarray: vector of evaluations of length N
        numpy.ndarray: n_rdm for each test_set
        numpy.ndarray: n_pattern for each test_set

    """
    models, evaluations, _, fitter = input_check_model(models, None, fitter, N)
    n_rdm = np.zeros(N, dtype=int)
    n_pattern = np.zeros(N, dtype=int)
    if pattern_descriptor is None:
        data.pattern_descriptors['index'] = np.arange(data.n_cond)
        pattern_descriptor = 'index'
    if rdm_descriptor is None:
        data.rdm_descriptors['index'] = np.arange(data.n_rdm)
        rdm_descriptor = 'index'
    for i_sample in range(N):
        sample, rdm_idx, pattern_idx = bootstrap_sample(
            data,
            rdm_descriptor=rdm_descriptor,
            pattern_descriptor=pattern_descriptor)
        train_set = [[sample, pattern_idx]]
        rdm_idx_test = data.rdm_descriptors[rdm_descriptor]
        rdm_idx_test = np.setdiff1d(rdm_idx_test, rdm_idx)
        pattern_idx_test = data.pattern_descriptors[pattern_descriptor]
        pattern_idx_test = np.setdiff1d(pattern_idx_test, pattern_idx)
        if len(pattern_idx_test) >= 3 and len(rdm_idx_test) >= 1:
            rdms_test = data.subsample_pattern(pattern_descriptor,
                                               pattern_idx_test)
            rdms_test = rdms_test.subsample(rdm_descriptor, rdm_idx_test)
            test_set = [[rdms_test, pattern_idx_test]]
            evaluations[i_sample] = crossval(
                models, data, train_set, test_set,
                method=method, fitter=fitter,
                pattern_descriptor=pattern_descriptor).evaluations[:, 0]
        else:
            evaluations[i_sample] = np.nan
        n_rdm[i_sample] = len(rdm_idx_test)
        n_pattern[i_sample] = len(pattern_idx_test)
    return evaluations, n_rdm, n_pattern


def bootstrap_testset_pattern(models, data, method='cosine', fitter=None,
                              N=1000, pattern_descriptor=None):
    """takes a bootstrap sample and evaluates on the patterns not
    sampled
    also returns the size of each test_set to allow later weighting
    or selection if this is desired.

    Args:
        models(rsatoolbox.model.Model): Model to be evaluated
        datat(rsatoolbox.rdm.RDMs): RDM data to use
        method(string): comparison method to use
        fitter(function): fitting function for the model
        pattern_descriptor(string): descriptor to group patterns

    Returns:
        numpy.ndarray: vector of evaluations of length
        numpy.ndarray: n_pattern for each test_set

    """
    models, evaluations, _, fitter = input_check_model(models, None, fitter, N)
    n_pattern = np.zeros(N, dtype=int)
    if pattern_descriptor is None:
        data.pattern_descriptors['index'] = np.arange(data.n_cond)
        pattern_descriptor = 'index'
    for i_sample in range(N):
        sample, pattern_idx = bootstrap_sample_pattern(
            data, pattern_descriptor=pattern_descriptor)
        train_set = [[sample, pattern_idx]]
        pattern_idx_test = data.pattern_descriptors[pattern_descriptor]
        pattern_idx_test = np.setdiff1d(pattern_idx_test, pattern_idx)
        if len(pattern_idx_test) >= 3:
            rdms_test = data.subsample_pattern(pattern_descriptor,
                                               pattern_idx_test)
            test_set = [[rdms_test, pattern_idx_test]]
            evaluations[i_sample] = crossval(
                models, data, train_set, test_set,
                method=method, fitter=fitter,
                pattern_descriptor=pattern_descriptor).evaluations[:, 0]
        else:
            evaluations[i_sample] = np.nan
        n_pattern[i_sample] = len(pattern_idx_test)
    return evaluations, n_pattern


def bootstrap_testset_rdm(models, data, method='cosine', fitter=None, N=1000,
                          rdm_descriptor=None):
    """takes a bootstrap sample and evaluates on the patterns not
    sampled
    also returns the size of each test_set to allow later weighting
    or selection if this is desired.

    Args:
        model(rsatoolbox.model.Model): Model to be evaluated
        datat(rsatoolbox.rdm.RDMs): RDM data to use
        method(string): comparison method to use
        fitter(function): fitting function for the model
        pattern_descriptor(string): descriptor to group patterns

    Returns:
        numpy.ndarray: vector of evaluations of length
        numpy.ndarray: n_pattern for each test_set

    """
    models, evaluations, _, fitter = input_check_model(models, None, fitter, N)
    n_rdm = np.zeros(N, dtype=int)
    if rdm_descriptor is None:
        data.rdm_descriptors['index'] = np.arange(data.n_rdm)
        rdm_descriptor = 'index'
    data.pattern_descriptors['index'] = np.arange(data.n_cond)
    pattern_descriptor = 'index'
    for i_sample in range(N):
        sample, rdm_idx = bootstrap_sample_rdm(
            data, rdm_descriptor=rdm_descriptor)
        pattern_idx = np.arange(data.n_cond)
        train_set = [[sample, pattern_idx]]
        rdm_idx_test = data.rdm_descriptors[rdm_descriptor]
        rdm_idx_test = np.setdiff1d(rdm_idx_test, rdm_idx)
        if len(rdm_idx_test) >= 1:
            rdms_test = data.subsample(rdm_descriptor, rdm_idx_test)
            test_set = [[rdms_test, pattern_idx]]
            evaluations[i_sample] = crossval(
                models, data, train_set, test_set,
                method=method, fitter=fitter,
                pattern_descriptor=pattern_descriptor).evaluations[:, 0]
        else:
            evaluations[i_sample] = np.nan
        n_rdm[i_sample] = len(rdm_idx_test)
    return evaluations, n_rdm
