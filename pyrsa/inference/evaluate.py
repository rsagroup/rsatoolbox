#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
inference module: evaluate models
@author: heiko
"""

import numpy as np
import tqdm
from collections.abc import Iterable
from pyrsa.rdm import compare
from pyrsa.inference import bootstrap_sample
from pyrsa.inference import bootstrap_sample_rdm
from pyrsa.inference import bootstrap_sample_pattern
from pyrsa.model import Model
from pyrsa.util.inference_util import input_check_model
from .result import Result
from .crossvalsets import sets_k_fold
from .noise_ceiling import boot_noise_ceiling
from .noise_ceiling import cv_noise_ceiling


def eval_fancy(model, data, method='cosine', fitter=None,
               k_pattern=5, k_rdm=5, N=1000,
               pattern_descriptor=None, rdm_descriptor=None):
    """evaluates a model by k-fold crossvalidation within a bootstrap
    Then uses the correction formula to get an estimate of the variance
    of the mean.

    If a k is set to 1 no crossvalidation is performed over the
    corresponding dimension.

    Args:
        model(pyrsa.model.Model): Model to be evaluated
        data(pyrsa.rdm.RDMs): RDM data to use
        method(string): comparison method to use
        fitter(function): fitting method for model
        k_pattern(int): #folds over patterns
        k_rdm(int): #folds over rdms
        N(int): number of bootstrap samples (default: 1000)
        pattern_descriptor(string): descriptor to group patterns
        rdm_descriptor(string): descriptor to group rdms
        random(bool): randomize group assignments (default: True)

    Returns:
        numpy.ndarray: matrix of evaluations (N x k)

    """
    result_full = bootstrap_crossval(
        model, data, method=method, fitter=fitter,
        k_pattern=k_pattern, k_rdm=k_rdm, N=N,
        pattern_descriptor=pattern_descriptor, rdm_descriptor=rdm_descriptor)
    result_rdm = bootstrap_crossval(
        model, data, method=method, fitter=fitter,
        k_pattern=k_pattern, k_rdm=k_rdm, N=N, boot_type='rdm',
        pattern_descriptor=pattern_descriptor, rdm_descriptor=rdm_descriptor)
    result_pattern = bootstrap_crossval(
        model, data, method=method, fitter=fitter,
        k_pattern=k_pattern, k_rdm=k_rdm, N=N, boot_type='pattern',
        pattern_descriptor=pattern_descriptor, rdm_descriptor=rdm_descriptor)
    eval_rdm = result_rdm.evaluations
    eval_rdm = eval_rdm[~np.isnan(eval_rdm[:, 0, 0])]
    eval_rdm = np.mean(eval_rdm, -1)
    var_rdm = np.cov(eval_rdm.T)
    eval_pattern = result_pattern.evaluations
    eval_pattern = eval_pattern[~np.isnan(eval_pattern[:, 0, 0])]
    eval_pattern = np.mean(eval_pattern, -1)
    var_pattern = np.cov(eval_pattern.T)
    eval_full = result_full.evaluations
    eval_full = eval_full[~np.isnan(eval_full[:, 0, 0])]
    eval_full = np.mean(eval_full, -1)
    var_full = np.cov(eval_full.T)
    var_estimate = 2 * (var_rdm + var_pattern) - var_full
    result = Result(model, result_full.evaluations, method=method,
                    cv_method='fancy', noise_ceiling=result_full.noise_ceiling,
                    variances=var_estimate)
    return result


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
    evaluations, theta, _ = input_check_model(model, theta, None, 1)
    if isinstance(model, Model):
        rdm_pred = model.predict_rdm(theta=theta)
        evaluations = np.array([[compare(rdm_pred, data, method)[0]]])
    elif isinstance(model, Iterable):
        evaluations = np.repeat(np.expand_dims(evaluations, -1),
                                data.n_rdm, -1)
        for k in range(len(model)):
            rdm_pred = model[k].predict_rdm(theta=theta[k])
            evaluations[k] = compare(rdm_pred, data, method)[0]
        evaluations = evaluations.reshape((1, len(model), data.n_rdm))
    else:
        raise ValueError('model should be a pyrsa.model.Model or a list of'
                         + ' such objects')
    noise_ceil = boot_noise_ceiling(
        data, method=method, rdm_descriptor='index')
    result = Result(model, evaluations, method=method,
                    cv_method='fixed', noise_ceiling=noise_ceil)
    return result


def eval_bootstrap(model, data, theta=None, method='cosine', N=1000,
                   pattern_descriptor=None, rdm_descriptor=None,
                   boot_noise_ceil=False):
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
    evaluations, theta, fitter = input_check_model(model, theta, None, N)
    noise_min = []
    noise_max = []
    for i in tqdm.trange(N):
        sample, rdm_sample, pattern_sample = \
            bootstrap_sample(data, rdm_descriptor=rdm_descriptor,
                             pattern_descriptor=pattern_descriptor)
        if isinstance(model, Model):
            rdm_pred = model.predict_rdm(theta=theta)
            rdm_pred = rdm_pred.subsample_pattern(pattern_descriptor,
                                                  pattern_sample)
            evaluations[i] = np.mean(compare(rdm_pred, sample, method))
        elif isinstance(model, Iterable):
            j = 0
            for mod in model:
                rdm_pred = mod.predict_rdm(theta=theta[j])
                rdm_pred = rdm_pred.subsample_pattern(pattern_descriptor,
                                                      pattern_sample)
                evaluations[i, j] = np.mean(compare(rdm_pred, sample, method))
                j += 1
        if boot_noise_ceil:
            noise_min_sample, noise_max_sample = boot_noise_ceiling(
                sample, method=method, rdm_descriptor=rdm_descriptor)
            noise_min.append(noise_min_sample)
            noise_max.append(noise_max_sample)
    if isinstance(model, Model):
        evaluations = evaluations.reshape((N, 1))
    if boot_noise_ceil:
        noise_ceil = np.array([noise_min, noise_max])
    else:
        noise_ceil = np.array(boot_noise_ceiling(
            data, method=method, rdm_descriptor=rdm_descriptor))
    result = Result(model, evaluations, method=method,
                    cv_method='bootstrap', noise_ceiling=noise_ceil)
    return result


def eval_bootstrap_pattern(model, data, theta=None, method='cosine', N=1000,
                           pattern_descriptor=None, rdm_descriptor=None,
                           boot_noise_ceil=True):
    """evaluates a model on data
    performs bootstrapping over patterns to get a sampling distribution

    Args:
        model(pyrsa.model.Model): Model to be evaluated
        data(pyrsa.rdm.RDMs): data to evaluate on
        theta(numpy.ndarray): parameter vector for the model
        method(string): comparison method to use
        N(int): number of samples
        pattern_descriptor(string): descriptor to group patterns for bootstrap
        rdm_descriptor(string): descriptor to group patterns for noise
            ceiling calculation

    Returns:
        numpy.ndarray: vector of evaluations

    """
    evaluations, theta, fitter = input_check_model(model, theta, None, N)
    noise_min = []
    noise_max = []
    for i in tqdm.trange(N):
        sample, pattern_sample = \
            bootstrap_sample_pattern(data, pattern_descriptor)
        if isinstance(model, Model):
            rdm_pred = model.predict_rdm(theta=theta)
            rdm_pred = rdm_pred.subsample_pattern(pattern_descriptor,
                                                  pattern_sample)
            evaluations[i] = np.mean(compare(rdm_pred, sample, method))
        elif isinstance(model, Iterable):
            j = 0
            for mod in model:
                rdm_pred = mod.predict_rdm(theta=theta[j])
                rdm_pred = rdm_pred.subsample_pattern(pattern_descriptor,
                                                      pattern_sample)
                evaluations[i, j] = np.mean(compare(rdm_pred, sample[0],
                                                    method))
                j += 1
        if boot_noise_ceil:
            noise_min_sample, noise_max_sample = boot_noise_ceiling(
                sample, method=method, rdm_descriptor=rdm_descriptor)
            noise_min.append(noise_min_sample)
            noise_max.append(noise_max_sample)
    if isinstance(model, Model):
        evaluations = evaluations.reshape((N, 1))
    if boot_noise_ceil:
        noise_ceil = np.array([noise_min, noise_max])
    else:
        noise_ceil = np.array(boot_noise_ceiling(
            data, method=method, rdm_descriptor=rdm_descriptor))
    result = Result(model, evaluations, method=method,
                    cv_method='bootstrap_pattern', noise_ceiling=noise_ceil)
    return result


def eval_bootstrap_rdm(model, data, theta=None, method='cosine', N=1000,
                       rdm_descriptor=None, boot_noise_ceil=False):
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
    evaluations, theta, _ = input_check_model(model, theta, None, N)
    noise_min = []
    noise_max = []
    for i in tqdm.trange(N):
        sample = bootstrap_sample_rdm(data, rdm_descriptor)
        if isinstance(model, Model):
            rdm_pred = model.predict_rdm(theta=theta)
            evaluations[i] = np.mean(compare(rdm_pred, sample[0], method))
        elif isinstance(model, Iterable):
            j = 0
            for mod in model:
                rdm_pred = mod.predict_rdm(theta=theta[j])
                evaluations[i, j] = np.mean(compare(rdm_pred, sample[0],
                                                    method))
                j += 1
        if boot_noise_ceil:
            noise_min_sample, noise_max_sample = boot_noise_ceiling(
                sample, method=method, rdm_descriptor=rdm_descriptor)
            noise_min.append(noise_min_sample)
            noise_max.append(noise_max_sample)
    if isinstance(model, Model):
        evaluations = evaluations.reshape((N, 1))
    if boot_noise_ceil:
        noise_ceil = np.array([noise_min, noise_max])
    else:
        noise_ceil = np.array(boot_noise_ceiling(
            data, method=method, rdm_descriptor=rdm_descriptor))
    result = Result(model, evaluations, method=method,
                    cv_method='bootstrap_rdm', noise_ceiling=noise_ceil)
    return result


def crossval(model, rdms, train_set, test_set, ceil_set=None, method='cosine',
             fitter=None, pattern_descriptor=None):
    """evaluates a model on cross-validation sets

    Args:
        model(pyrsa.model.Model): Model to be evaluated
        rdms(pyrsa.rdm.RDMs): full dataset
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
    if ceil_set is not None:
        assert len(ceil_set) == len(test_set), \
            'ceil_set and test_set must have the same length'
    if pattern_descriptor is None:
        pattern_descriptor = 'index'
    evaluations = []
    noise_ceil = []
    for i in range(len(train_set)):
        train = train_set[i]
        test = test_set[i]
        if (train[0].n_rdm == 0 or test[0].n_rdm == 0 or
                train[0].n_cond <= 2 or test[0].n_cond <= 2):
            if isinstance(model, Model):
                evals = np.nan
            elif isinstance(model, Iterable):
                evals = np.empty(len(model)) * np.nan
        else:
            if isinstance(model, Model):
                if fitter is None:
                    fitter = model.default_fitter
                theta = fitter(model, train[0], method=method,
                               pattern_sample=train[1],
                               pattern_descriptor=pattern_descriptor)
                pred = model.predict_rdm(theta)
                pred = pred.subsample_pattern(by=pattern_descriptor,
                                              value=test[1])
                evals = np.mean(compare(pred, test[0], method))
            elif isinstance(model, Iterable):
                evals, _, fitter = input_check_model(model, None, fitter)
                for j in range(len(model)):
                    theta = fitter[j](model[j], train[0], method=method,
                                      pattern_sample=train[1],
                                      pattern_descriptor=pattern_descriptor)
                    pred = model[j].predict_rdm(theta)
                    pred = pred.subsample_pattern(by=pattern_descriptor,
                                                  value=test[1])
                    evals[j] = np.mean(compare(pred, test[0], method))
            if ceil_set is None:
                noise_ceil.append(boot_noise_ceiling(
                    rdms.subsample_pattern(by=pattern_descriptor,
                                           value=test[1]),
                    method=method))
        evaluations.append(evals)
    if isinstance(model, Model):
        model = [model]
    evaluations = np.array(evaluations).T  # .T to switch model/set order
    evaluations = evaluations.reshape((1, len(model), len(train_set)))
    if ceil_set is not None:
        noise_ceil = cv_noise_ceiling(rdms, ceil_set, test_set, method=method,
                                      pattern_descriptor=pattern_descriptor)
    else:
        noise_ceil = np.array(noise_ceil).T
    result = Result(model, evaluations, method=method,
                    cv_method='crossvalidation', noise_ceiling=noise_ceil)
    return result


def bootstrap_crossval(model, data, method='cosine', fitter=None,
                       k_pattern=5, k_rdm=5, N=1000,
                       pattern_descriptor=None, rdm_descriptor=None,
                       random=True, boot_type='both'):
    """evaluates a model by k-fold crossvalidation within a bootstrap

    If a k is set to 1 no crossvalidation is performed over the
    corresponding dimension.


    Args:
        model(pyrsa.model.Model): Model to be evaluated
        data(pyrsa.rdm.RDMs): RDM data to use
        method(string): comparison method to use
        fitter(function): fitting method for model
        k_pattern(int): #folds over patterns
        k_rdm(int): #folds over rdms
        N(int): number of bootstrap samples (default: 1000)
        pattern_descriptor(string): descriptor to group patterns
        rdm_descriptor(string): descriptor to group rdms
        random(bool): randomize group assignments (default: True)
        boot_type(String): which dimension to bootstrap over (default: 'both')
            alternatives: 'rdm', 'pattern'

    Returns:
        numpy.ndarray: matrix of evaluations (N x k)

    """
    if rdm_descriptor is None:
        rdm_select = np.arange(data.n_rdm)
        data.rdm_descriptors['index'] = rdm_select
        rdm_descriptor = 'index'
    if pattern_descriptor is None:
        pattern_select = np.arange(data.n_cond)
        data.pattern_descriptors['index'] = pattern_select
        pattern_descriptor = 'index'
    if isinstance(model, Model):
        evaluations = np.zeros((N, 1, k_pattern * k_rdm))
    elif isinstance(model, Iterable):
        evaluations = np.zeros((N, len(model), k_pattern * k_rdm))
    noise_ceil = np.zeros((2, N))
    for i_sample in tqdm.trange(N):
        if boot_type == 'both':
            sample, rdm_sample, pattern_sample = bootstrap_sample(
                data,
                rdm_descriptor=rdm_descriptor,
                pattern_descriptor=pattern_descriptor)
        elif boot_type == 'pattern':
            sample, pattern_sample = bootstrap_sample_pattern(
                data,
                pattern_descriptor=pattern_descriptor)
            rdm_sample = np.unique(data.rdm_descriptors[rdm_descriptor])
        elif boot_type == 'rdm':
            sample, rdm_sample = bootstrap_sample_rdm(
                data,
                rdm_descriptor=rdm_descriptor)
            pattern_sample = np.unique(
                data.pattern_descriptors[pattern_descriptor])
        else:
            raise ValueError('boot_type not understood')
        if len(np.unique(rdm_sample)) >= k_rdm \
           and len(np.unique(pattern_sample)) >= 3 * k_pattern:
            train_set, test_set, ceil_set = sets_k_fold(
                sample,
                pattern_descriptor=pattern_descriptor,
                rdm_descriptor=rdm_descriptor,
                k_pattern=k_pattern, k_rdm=k_rdm, random=random)
            for idx in range(len(test_set)):
                test_set[idx][1] = _concat_sampling(pattern_sample,
                                                    test_set[idx][1])
                train_set[idx][1] = _concat_sampling(pattern_sample,
                                                     train_set[idx][1])
            cv_result = crossval(
                model, sample,
                train_set, test_set,
                method=method, fitter=fitter,
                pattern_descriptor=pattern_descriptor)
            if isinstance(model, Model):
                evaluations[i_sample, 0, :] = cv_result.evaluations[0, 0]
            elif isinstance(model, Iterable):
                evaluations[i_sample, :, :] = cv_result.evaluations[0]
            noise_ceil[:, i_sample] = np.mean(cv_result.noise_ceiling, axis=-1)
        else:  # sample does not allow desired crossvalidation
            if isinstance(model, Model):
                evaluations[i_sample, 0, :] = np.nan
            elif isinstance(model, Iterable):
                evaluations[i_sample, :, :] = np.nan
            noise_ceil[:, i_sample] = np.nan
    result = Result(model, evaluations, method=method,
                    cv_method='bootstrap_crossval', noise_ceiling=noise_ceil)
    return result


def _concat_sampling(sample1, sample2):
    """ computes an index vector for the sequential sampling with sample1
    and sample2
    """
    sample_out = [[i_samp1 for i_samp1 in sample1 if i_samp1 == i_samp2]
                  for i_samp2 in sample2]
    return sum(sample_out, [])
