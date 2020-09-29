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
from pyrsa.util.inference_util import default_k_pattern, default_k_rdm
from .result import Result
from .crossvalsets import sets_k_fold
from .noise_ceiling import boot_noise_ceiling
from .noise_ceiling import cv_noise_ceiling


def eval_fancy(models, data, method='cosine', fitter=None,
               k_pattern=None, k_rdm=None, N=1000, boot_noise_ceil=False,
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
        models, data, method=method, fitter=fitter,
        k_pattern=k_pattern, k_rdm=k_rdm, N=N,
        pattern_descriptor=pattern_descriptor, rdm_descriptor=rdm_descriptor)
    result_rdm = bootstrap_crossval(
        models, data, method=method, fitter=fitter,
        k_pattern=k_pattern, k_rdm=k_rdm, N=N, boot_type='rdm',
        pattern_descriptor=pattern_descriptor, rdm_descriptor=rdm_descriptor)
    result_pattern = bootstrap_crossval(
        models, data, method=method, fitter=fitter,
        k_pattern=k_pattern, k_rdm=k_rdm, N=N, boot_type='pattern',
        pattern_descriptor=pattern_descriptor, rdm_descriptor=rdm_descriptor)
    eval_rdm = result_rdm.evaluations
    ok_rdm = ~np.isnan(eval_rdm[:, 0, 0])
    eval_rdm = eval_rdm[ok_rdm]
    nc_rdm = result_rdm.noise_ceiling[:, ok_rdm]
    eval_rdm = np.mean(eval_rdm, -1)
    var_rdm = np.cov(np.concatenate([eval_rdm.T, nc_rdm]))
    eval_pattern = result_pattern.evaluations
    ok_pattern = ~np.isnan(eval_pattern[:, 0, 0])
    eval_pattern = eval_pattern[ok_pattern]
    nc_pattern = result_pattern.noise_ceiling[:, ok_pattern]
    eval_pattern = np.mean(eval_pattern, -1)
    var_pattern = np.cov(np.concatenate([eval_pattern.T, nc_pattern]))
    eval_full = result_full.evaluations
    ok_full = ~np.isnan(eval_full[:, 0, 0])
    eval_full = eval_full[ok_full]
    nc_full = result_full.noise_ceiling[:, ok_full]
    eval_full = np.mean(eval_full, -1)
    var_full = np.cov(np.concatenate([eval_full.T, nc_full]))
    var_estimate = 2 * (var_rdm + var_pattern) - var_full
    result = Result(models, result_full.evaluations, method=method,
                    cv_method='fancy',
                    noise_ceiling=result_full.noise_ceiling,
                    variances=var_estimate[:-2, :-2],
                    noise_ceil_var=var_estimate[:, -2:],
                    dof=result_full.dof)
    return result


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
        evaluations = np.repeat(np.expand_dims(evaluations, -1),
                                data.n_rdm, -1)
        for k in range(len(models)):
            rdm_pred = models[k].predict_rdm(theta=theta[k])
            evaluations[k] = compare(rdm_pred, data, method)
        evaluations = evaluations.reshape((1, len(models), data.n_rdm))
    else:
        raise ValueError('models should be a pyrsa.model.Model or a list of'
                         + ' such objects')
    noise_ceil = boot_noise_ceiling(
        data, method=method, rdm_descriptor='index')
    variances = np.cov(evaluations[0], ddof=1) \
        / evaluations.shape[-1]
    noise_ceil_var = np.zeros((evaluations.shape[1] + 2, 2))
    dof = evaluations.shape[-1] - 1
    result = Result(models, evaluations, method=method,
                    cv_method='fixed', noise_ceiling=noise_ceil,
                    variances=variances, dof=dof,
                    noise_ceil_var=noise_ceil_var)
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
        var = np.cov(np.concatenate([evaluations.T, noise_ceil]))
        variances = var[:-2, :-2]
        noise_ceil_var = var[:, -2:]
    else:
        noise_ceil = np.array(boot_noise_ceiling(
            data, method=method, rdm_descriptor=rdm_descriptor))
        variances = np.cov(evaluations.T)
        noise_ceil_var = None
    dof = min(data.n_rdm, data.n_cond) - 1
    result = Result(models, evaluations, method=method,
                    cv_method='bootstrap', noise_ceiling=noise_ceil,
                    variances=variances, dof=dof,
                    noise_ceil_var=noise_ceil_var)
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
        var = np.cov(np.concatenate([evaluations.T, noise_ceil]))
        variances = var[:-2, :-2]
        noise_ceil_var = var[:, -2:]
    else:
        noise_ceil = np.array(boot_noise_ceiling(
            data, method=method, rdm_descriptor=rdm_descriptor))
        variances = np.cov(evaluations.T)
        noise_ceil_var = None
    dof = data.n_cond - 1
    result = Result(models, evaluations, method=method,
                    cv_method='bootstrap_pattern', noise_ceiling=noise_ceil,
                    variances=variances, dof=dof,
                    noise_ceil_var=noise_ceil_var)
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
        var = np.cov(np.concatenate([evaluations.T, noise_ceil]))
        variances = var[:-2, :-2]
        noise_ceil_var = var[:, -2:]
    else:
        noise_ceil = np.array(boot_noise_ceiling(
            data, method=method, rdm_descriptor=rdm_descriptor))
        variances = np.cov(evaluations.T)
        noise_ceil_var = None
    dof = data.n_rdm - 1
    variances = np.cov(evaluations.T)
    result = Result(models, evaluations, method=method,
                    cv_method='bootstrap_rdm', noise_ceiling=noise_ceil,
                    variances=variances, dof=dof,
                    noise_ceil_var=noise_ceil_var)
    return result


def crossval(models, rdms, train_set, test_set, ceil_set=None, method='cosine',
             fitter=None, pattern_descriptor='index', calc_noise_ceil=True):
    """evaluates a model on cross-validation sets

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
            if ceil_set is None and calc_noise_ceil:
                noise_ceil.append(boot_noise_ceiling(
                    rdms.subsample_pattern(by=pattern_descriptor,
                                           value=test[1]), method=method))
        evaluations.append(evals)
    if isinstance(models, Model):
        models = [models]
    evaluations = np.array(evaluations).T  # .T to switch model/set order
    evaluations = evaluations.reshape((1, len(models), len(train_set)))
    if ceil_set is not None and calc_noise_ceil:
        noise_ceil = cv_noise_ceiling(rdms, ceil_set, test_set, method=method,
                                      pattern_descriptor=pattern_descriptor)
    elif calc_noise_ceil:
        noise_ceil = np.array(noise_ceil).T
    else:
        noise_ceil = np.array([np.nan, np.nan])
    result = Result(models, evaluations, method=method,
                    cv_method='crossvalidation', noise_ceiling=noise_ceil)
    return result


def bootstrap_crossval(models, data, method='cosine', fitter=None,
                       k_pattern=None, k_rdm=None, N=1000,
                       pattern_descriptor='index', rdm_descriptor='index',
                       random=True, boot_type='both'):
    """evaluates a model by k-fold crossvalidation within a bootstrap

    If a k is set to 1 no crossvalidation is performed over the
    corresponding dimension.

    by default ks are set by pyrsa.util.inference_util.default_k_pattern
    and pyrsa.util.inference_util.default_k_rdm based on the number of
    rdms and patterns provided. the ks are then in the range 2-5.

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
        boot_type(String): which dimension to bootstrap over (default: 'both')
            alternatives: 'rdm', 'pattern'

    Returns:
        numpy.ndarray: matrix of evaluations (N x k)

    """
    if k_pattern is None:
        n_pattern = len(np.unique(data.pattern_descriptors[
            pattern_descriptor]))
        k_pattern = default_k_pattern((1 - 1 / np.exp(1)) * n_pattern)
    if k_rdm is None:
        n_rdm = len(np.unique(data.rdm_descriptors[
            rdm_descriptor]))
        k_rdm = default_k_rdm((1 - 1 / np.exp(1)) * n_rdm)
    if isinstance(models, Model):
        models = [models]
    evaluations = np.zeros((N, len(models), k_pattern * k_rdm))
    noise_ceil = np.zeros((2, N))
    for i_sample in tqdm.trange(N):
        if boot_type == 'both':
            sample, rdm_idx, pattern_idx = bootstrap_sample(
                data,
                rdm_descriptor=rdm_descriptor,
                pattern_descriptor=pattern_descriptor)
        elif boot_type == 'pattern':
            sample, pattern_idx = bootstrap_sample_pattern(
                data,
                pattern_descriptor=pattern_descriptor)
            rdm_idx = np.unique(data.rdm_descriptors[rdm_descriptor])
        elif boot_type == 'rdm':
            sample, rdm_idx = bootstrap_sample_rdm(
                data,
                rdm_descriptor=rdm_descriptor)
            pattern_idx = np.unique(
                data.pattern_descriptors[pattern_descriptor])
        else:
            raise ValueError('boot_type not understood')
        if len(np.unique(rdm_idx)) >= k_rdm \
           and len(np.unique(pattern_idx)) >= 3 * k_pattern:
            train_set, test_set, ceil_set = sets_k_fold(
                sample,
                pattern_descriptor=pattern_descriptor,
                rdm_descriptor=rdm_descriptor,
                k_pattern=k_pattern, k_rdm=k_rdm, random=random)
            if k_rdm > 1 or k_pattern > 1:
                cv_nc = cv_noise_ceiling(sample, ceil_set, test_set,
                                         method=method,
                                         pattern_descriptor=pattern_descriptor)
                noise_ceil[:, i_sample] = cv_nc
            else:
                nc = boot_noise_ceiling(
                    sample,
                    method=method,
                    rdm_descriptor=rdm_descriptor)
                noise_ceil[:, i_sample] = nc
            for idx in range(len(test_set)):
                test_set[idx][1] = _concat_sampling(pattern_idx,
                                                    test_set[idx][1])
                train_set[idx][1] = _concat_sampling(pattern_idx,
                                                     train_set[idx][1])
            cv_result = crossval(
                models, sample,
                train_set, test_set,
                method=method, fitter=fitter,
                pattern_descriptor=pattern_descriptor,
                calc_noise_ceil=False)
            evaluations[i_sample, :, :] = cv_result.evaluations[0]
            noise_ceil[:, i_sample] = np.mean(cv_result.noise_ceiling, axis=-1)
        else:  # sample does not allow desired crossvalidation
            evaluations[i_sample, :, :] = np.nan
            noise_ceil[:, i_sample] = np.nan
    if boot_type == 'both':
        cv_method = 'bootstrap_crossval'
        dof = min(data.n_rdm, data.n_cond) - 1
    elif boot_type == 'pattern':
        cv_method = 'bootstrap_crossval_pattern'
        dof = data.n_cond - 1
    elif boot_type == 'rdm':
        cv_method = 'bootstrap_crossval_rdm'
        dof = data.n_rdm - 1
    eval_ok = ~np.isnan(evaluations[:, 0, 0])
    evals_nonan = np.mean(evaluations[eval_ok], -1)
    noise_ceil_nonan = noise_ceil[:, eval_ok]
    variances = np.cov(np.concatenate([evals_nonan.T, noise_ceil_nonan]))
    result = Result(models, evaluations, method=method,
                    cv_method=cv_method, noise_ceiling=noise_ceil,
                    variances=variances[:-2, :-2], dof=dof,
                    noise_ceil_var=variances[:, -2:])
    return result


def _concat_sampling(sample1, sample2):
    """ computes an index vector for the sequential sampling with sample1
    and sample2
    """
    sample_out = [[i_samp1 for i_samp1 in sample1 if i_samp1 == i_samp2]
                  for i_samp2 in sample2]
    return sum(sample_out, [])
