#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate model performance
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


def eval_fixed(models, data, theta=None, method='cosine'):
    """evaluates models on data, without any bootstrapping or
    cross-validation

    Args:
        models(list of pyrsa.model.Model): models to be evaluated
        data(pyrsa.rdm.RDMs): data to evaluate on
        theta(numpy.ndarray): parameter vector for the models
        method(string): comparison method to use

    Returns:
        float: evaluation

    """
    evaluations, theta, _ = input_check_model(models, theta, None, 1)
    if isinstance(models, Model):
        rdm_pred = models.predict_rdm(theta=theta)
        evaluations = np.array([[compare(rdm_pred, data, method)[0]]])
    elif isinstance(models, Iterable):
        for k in range(len(models)):
            rdm_pred = models[k].predict_rdm(theta=theta[k])
            evaluations[k] = np.mean(compare(rdm_pred, data, method)[0])
        evaluations = evaluations.reshape((1, len(models)))
    else:
        raise ValueError('models should be a pyrsa.model.Model or a list of'
                         + ' such objects')
    noise_ceil = boot_noise_ceiling(
        data, method=method, rdm_descriptor='index')
    result = Result(models, evaluations, method=method,
                    cv_method='fixed', noise_ceiling=noise_ceil)
    return result


def eval_bootstrap(models, data, theta=None, method='cosine', N=1000,
                   pattern_descriptor='index', rdm_descriptor='index',
                   boot_noise_ceil=True):
    """evaluates models on data
    performs bootstrapping to get a sampling distribution

    Args:
        models(pyrsa.model.Model): models to be evaluated
        data(pyrsa.rdm.RDMs): data to evaluate on
        theta(numpy.ndarray): parameter vector for the models
        method(string): comparison method to use
        N(int): number of samples
        pattern_descriptor(string): descriptor to group patterns for bootstrap
        rdm_descriptor(string): descriptor to group rdms for bootstrap

    Returns:
        numpy.ndarray: vector of evaluations

    """
    evaluations, theta, fitter = input_check_model(models, theta, None, N)
    noise_min = []
    noise_max = []
    for i in tqdm.trange(N):
        sample, rdm_idx, pattern_idx = \
            bootstrap_sample(data, rdm_descriptor=rdm_descriptor,
                             pattern_descriptor=pattern_descriptor)
        if len(np.unique(pattern_idx)) >= 3:
            if isinstance(models, Model):
                rdm_pred = models.predict_rdm(theta=theta)
                rdm_pred = rdm_pred.subsample_pattern(pattern_descriptor,
                                                      pattern_idx)
                evaluations[i] = np.mean(compare(rdm_pred, sample, method))
            elif isinstance(models, Iterable):
                j = 0
                for mod in models:
                    rdm_pred = mod.predict_rdm(theta=theta[j])
                    rdm_pred = rdm_pred.subsample_pattern(pattern_descriptor,
                                                          pattern_idx)
                    evaluations[i, j] = np.mean(compare(rdm_pred, sample,
                                                        method))
                    j += 1
            if boot_noise_ceil:
                noise_min_sample, noise_max_sample = boot_noise_ceiling(
                    sample, method=method, rdm_descriptor=rdm_descriptor)
                noise_min.append(noise_min_sample)
                noise_max.append(noise_max_sample)
        else:
            if isinstance(models, Model):
                evaluations[i] = np.nan
            elif isinstance(models, Iterable):
                evaluations[i, :] = np.nan
            noise_min.append(np.nan)
            noise_max.append(np.nan)
    if isinstance(models, Model):
        evaluations = evaluations.reshape((N, 1))
    if boot_noise_ceil:
        noise_ceil = np.array([noise_min, noise_max])
    else:
        noise_ceil = np.array(boot_noise_ceiling(
            data, method=method, rdm_descriptor=rdm_descriptor))
    result = Result(models, evaluations, method=method,
                    cv_method='bootstrap', noise_ceiling=noise_ceil)
    return result


def eval_bootstrap_pattern(models, data, theta=None, method='cosine', N=1000,
                           pattern_descriptor='index', rdm_descriptor='index',
                           boot_noise_ceil=True):
    """evaluates a models on data
    performs bootstrapping over patterns to get a sampling distribution

    Args:
        models(pyrsa.model.Model): models to be evaluated
        data(pyrsa.rdm.RDMs): data to evaluate on
        theta(numpy.ndarray): parameter vector for the models
        method(string): comparison method to use
        N(int): number of samples
        pattern_descriptor(string): descriptor to group patterns for bootstrap
        rdm_descriptor(string): descriptor to group patterns for noise
            ceiling calculation

    Returns:
        numpy.ndarray: vector of evaluations

    """
    evaluations, theta, fitter = input_check_model(models, theta, None, N)
    noise_min = []
    noise_max = []
    for i in tqdm.trange(N):
        sample, pattern_idx = \
            bootstrap_sample_pattern(data, pattern_descriptor)
        if len(np.unique(pattern_idx)) >= 3:
            if isinstance(models, Model):
                rdm_pred = models.predict_rdm(theta=theta)
                rdm_pred = rdm_pred.subsample_pattern(pattern_descriptor,
                                                      pattern_idx)
                evaluations[i] = np.mean(compare(rdm_pred, sample, method))
            elif isinstance(models, Iterable):
                j = 0
                for mod in models:
                    rdm_pred = mod.predict_rdm(theta=theta[j])
                    rdm_pred = rdm_pred.subsample_pattern(pattern_descriptor,
                                                          pattern_idx)
                    evaluations[i, j] = np.mean(compare(rdm_pred, sample,
                                                        method))
                    j += 1
            if boot_noise_ceil:
                noise_min_sample, noise_max_sample = boot_noise_ceiling(
                    sample, method=method, rdm_descriptor=rdm_descriptor)
                noise_min.append(noise_min_sample)
                noise_max.append(noise_max_sample)
        else:
            if isinstance(models, Model):
                evaluations[i] = np.nan
            elif isinstance(models, Iterable):
                evaluations[i, :] = np.nan
            noise_min.append(np.nan)
            noise_max.append(np.nan)
    if isinstance(models, Model):
        evaluations = evaluations.reshape((N, 1))
    if boot_noise_ceil:
        noise_ceil = np.array([noise_min, noise_max])
    else:
        noise_ceil = np.array(boot_noise_ceiling(
            data, method=method, rdm_descriptor=rdm_descriptor))
    result = Result(models, evaluations, method=method,
                    cv_method='bootstrap_pattern', noise_ceiling=noise_ceil)
    return result


def eval_bootstrap_rdm(models, data, theta=None, method='cosine', N=1000,
                       rdm_descriptor='index', boot_noise_ceil=True):
    """evaluates models on data
    performs bootstrapping to get a sampling distribution

    Args:
        models(pyrsa.model.Model): models to be evaluated
        data(pyrsa.rdm.RDMs): data to evaluate on
        theta(numpy.ndarray): parameter vector for the models
        method(string): comparison method to use
        N(int): number of samples
        rdm_descriptor(string): rdm_descriptor to group rdms for bootstrap

    Returns:
        numpy.ndarray: vector of evaluations

    """
    evaluations, theta, _ = input_check_model(models, theta, None, N)
    noise_min = []
    noise_max = []
    for i in tqdm.trange(N):
        sample, rdm_idx = bootstrap_sample_rdm(data, rdm_descriptor)
        if isinstance(models, Model):
            rdm_pred = models.predict_rdm(theta=theta)
            evaluations[i] = np.mean(compare(rdm_pred, sample, method))
        elif isinstance(models, Iterable):
            j = 0
            for mod in models:
                rdm_pred = mod.predict_rdm(theta=theta[j])
                evaluations[i, j] = np.mean(compare(rdm_pred, sample,
                                                    method))
                j += 1
        if boot_noise_ceil:
            noise_min_sample, noise_max_sample = boot_noise_ceiling(
                sample, method=method, rdm_descriptor=rdm_descriptor)
            noise_min.append(noise_min_sample)
            noise_max.append(noise_max_sample)
    if isinstance(models, Model):
        evaluations = evaluations.reshape((N, 1))
    if boot_noise_ceil:
        noise_ceil = np.array([noise_min, noise_max])
    else:
        noise_ceil = np.array(boot_noise_ceiling(
            data, method=method, rdm_descriptor=rdm_descriptor))
    result = Result(models, evaluations, method=method,
                    cv_method='bootstrap_rdm', noise_ceiling=noise_ceil)
    return result


def crossval(models, rdms, train_set, test_set, ceil_set=None, method='cosine',
             fitter=None, pattern_descriptor='index'):
    """evaluates models on cross-validation sets

    Args:
        models(pyrsa.model.Model): models to be evaluated
        rdms(pyrsa.rdm.RDMs): full dataset
        train_set(list): a list of the training RDMs with 2-tuple entries:
            (RDMs, pattern_idx)
        test_set(list): a list of the test RDMs with 2-tuple entries:
            (RDMs, pattern_idx)
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
    evaluations = []
    noise_ceil = []
    for i in range(len(train_set)):
        train = train_set[i]
        test = test_set[i]
        if (train[0].n_rdm == 0 or test[0].n_rdm == 0 or
                train[0].n_cond <= 2 or test[0].n_cond <= 2):
            if isinstance(models, Model):
                evals = np.nan
            elif isinstance(models, Iterable):
                evals = np.empty(len(models)) * np.nan
        else:
            if isinstance(models, Model):
                if fitter is None:
                    fitter = models.default_fitter
                theta = fitter(models, train[0], method=method,
                               pattern_idx=train[1],
                               pattern_descriptor=pattern_descriptor)
                pred = models.predict_rdm(theta)
                pred = pred.subsample_pattern(by=pattern_descriptor,
                                              value=test[1])
                evals = np.mean(compare(pred, test[0], method))
            elif isinstance(models, Iterable):
                evals, _, fitter = input_check_model(models, None, fitter)
                for j in range(len(models)):
                    theta = fitter[j](models[j], train[0], method=method,
                                      pattern_idx=train[1],
                                      pattern_descriptor=pattern_descriptor)
                    pred = models[j].predict_rdm(theta)
                    pred = pred.subsample_pattern(by=pattern_descriptor,
                                                  value=test[1])
                    evals[j] = np.mean(compare(pred, test[0], method))
            if ceil_set is None:
                noise_ceil.append(boot_noise_ceiling(
                    rdms.subsample_pattern(by=pattern_descriptor,
                                           value=test[1]),
                    method=method))
        evaluations.append(evals)
    if isinstance(models, Model):
        models = [models]
    evaluations = np.array(evaluations).T  # .T to switch models/set order
    evaluations = evaluations.reshape((1, len(models), len(train_set)))
    if ceil_set is not None:
        noise_ceil = cv_noise_ceiling(rdms, ceil_set, test_set, method=method,
                                      pattern_descriptor=pattern_descriptor)
    else:
        noise_ceil = np.array(noise_ceil).T
    result = Result(models, evaluations, method=method,
                    cv_method='crossvalidation', noise_ceiling=noise_ceil)
    return result


def bootstrap_crossval(models, data, method='cosine', fitter=None,
                       k_pattern=5, k_rdm=5, N=1000,
                       pattern_descriptor='index', rdm_descriptor='index',
                       random=True):
    """evaluates models by k-fold crossvalidation within a bootstrap

    If a k is set to 1 no crossvalidation is performed over the
    corresponding dimension.


    Args:
        models(pyrsa.model.Model): models to be evaluated
        data(pyrsa.rdm.RDMs): RDM data to use
        method(string): comparison method to use
        fitter(function): fitting method for models
        k_pattern(int): #folds over patterns
        k_rdm(int): #folds over rdms
        N(int): number of bootstrap samples (default: 1000)
        pattern_descriptor(string): descriptor to group patterns
        rdm_descriptor(string): descriptor to group rdms
        random(bool): randomize group assignments (default: True)

    Returns:
        numpy.ndarray: matrix of evaluations (N x k)

    """
    if isinstance(models, Model):
        evaluations = np.zeros((N, 1, k_pattern * k_rdm))
    elif isinstance(models, Iterable):
        evaluations = np.zeros((N, len(models), k_pattern * k_rdm))
    noise_ceil = np.zeros((2, N))
    for i_sample in tqdm.trange(N):
        sample, rdm_idx, pattern_idx = bootstrap_sample(
            data,
            rdm_descriptor=rdm_descriptor,
            pattern_descriptor=pattern_descriptor)
        if len(np.unique(rdm_idx)) >= k_rdm \
           and len(np.unique(pattern_idx)) >= 3 * k_pattern:
            train_set, test_set, ceil_set = sets_k_fold(
                sample,
                pattern_descriptor=pattern_descriptor,
                rdm_descriptor=rdm_descriptor,
                k_pattern=k_pattern, k_rdm=k_rdm, random=random)
            for idx in range(len(test_set)):
                test_set[idx][1] = _concat_sampling(pattern_idx,
                                                    test_set[idx][1])
                train_set[idx][1] = _concat_sampling(pattern_idx,
                                                     train_set[idx][1])
            cv_result = crossval(
                models, sample,
                train_set, test_set,
                method=method, fitter=fitter,
                pattern_descriptor=pattern_descriptor)
            if isinstance(models, Model):
                evaluations[i_sample, 0, :] = cv_result.evaluations[0, 0]
            elif isinstance(models, Iterable):
                evaluations[i_sample, :, :] = cv_result.evaluations[0]
            noise_ceil[:, i_sample] = np.mean(cv_result.noise_ceiling, axis=-1)
        else:  # sample does not allow desired crossvalidation
            if isinstance(models, Model):
                evaluations[i_sample, 0, :] = np.nan
            elif isinstance(models, Iterable):
                evaluations[i_sample, :, :] = np.nan
            noise_ceil[:, i_sample] = np.nan
    result = Result(models, evaluations, method=method,
                    cv_method='bootstrap_crossval', noise_ceiling=noise_ceil)
    return result


def _concat_sampling(sample1, sample2):
    """ computes an index vector for the sequential sampling with sample1
    and sample2
    """
    sample_out = [[i_samp1 for i_samp1 in sample1 if i_samp1 == i_samp2]
                  for i_samp2 in sample2]
    return sum(sample_out, [])
