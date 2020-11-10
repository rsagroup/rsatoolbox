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


def eval_fancy(models, data, method='cosine', fitter=None, n_cv=2,
               k_pattern=None, k_rdm=None, N=1000, boot_noise_ceil=False,
               pattern_descriptor='index', rdm_descriptor='index',
               use_correction=True):
    """evaluates a model by k-fold crossvalidation within a bootstrap
    Then uses the correction formula to get an estimate of the variance
    of the mean.

    If a k is set to 1 no crossvalidation is performed over the
    corresponding dimension.

    Args:
        models(pyrsa.model.Model or list): Models to be evaluated
        data(pyrsa.rdm.RDMs): RDM data to use
        method(string): comparison method to use
        fitter(function): fitting method for models
        n_cv(int): number of crossvalidation runs per sample
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
        pattern_descriptor=pattern_descriptor, rdm_descriptor=rdm_descriptor,
        n_cv=n_cv, use_correction=use_correction)
    result_rdm = bootstrap_crossval(
        models, data, method=method, fitter=fitter,
        k_pattern=k_pattern, k_rdm=k_rdm, N=N, boot_type='rdm',
        pattern_descriptor=pattern_descriptor, rdm_descriptor=rdm_descriptor,
        n_cv=n_cv, use_correction=use_correction)
    result_pattern = bootstrap_crossval(
        models, data, method=method, fitter=fitter,
        k_pattern=k_pattern, k_rdm=k_rdm, N=N, boot_type='pattern',
        pattern_descriptor=pattern_descriptor, rdm_descriptor=rdm_descriptor,
        n_cv=n_cv, use_correction=use_correction)
    var_estimate = 2 * (result_rdm.variances + result_pattern.variances) \
        - result_full.variances
    if result_rdm.noise_ceil_var is not None \
        and result_pattern.noise_ceil_var is not None \
        and result_full.noise_ceil_var is not None:
        var_nc_estimate = 2 * (result_rdm.noise_ceil_var
                               + result_pattern.noise_ceil_var) \
            - result_full.noise_ceil_var
    else:
        var_nc_estimate = None
    result = Result(models, result_full.evaluations, method=method,
                    cv_method='fancy',
                    noise_ceiling=result_full.noise_ceiling,
                    variances=var_estimate,
                    noise_ceil_var=var_nc_estimate,
                    dof=result_full.dof)
    return result


def eval_fixed(models, data, theta=None, method='cosine'):
    """evaluates models on data, without any bootstrapping or
    cross-validation

    Args:
        models(list of pyrsa.model.Model or list): models to be evaluated
        data(pyrsa.rdm.RDMs): data to evaluate on
        theta(numpy.ndarray): parameter vector for the models
        method(string): comparison method to use

    Returns:
        float: evaluation

    """
    models, evaluations, theta, _ = input_check_model(models, theta, None, 1)
    evaluations = np.repeat(np.expand_dims(evaluations, -1),
                            data.n_rdm, -1)
    for k in range(len(models)):
        rdm_pred = models[k].predict_rdm(theta=theta[k])
        evaluations[k] = compare(rdm_pred, data, method)
    evaluations = evaluations.reshape((1, len(models), data.n_rdm))
    noise_ceil = boot_noise_ceiling(
        data, method=method, rdm_descriptor='index')
    variances = np.cov(evaluations[0], ddof=1) \
        / evaluations.shape[-1]
    noise_ceil_var = None
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
        models(pyrsa.model.Model or list): models to be evaluated
        data(pyrsa.rdm.RDMs): data to evaluate on
        theta(numpy.ndarray): parameter vector for the models
        method(string): comparison method to use
        N(int): number of samples
        pattern_descriptor(string): descriptor to group patterns for bootstrap
        rdm_descriptor(string): descriptor to group rdms for bootstrap

    Returns:
        numpy.ndarray: vector of evaluations

    """
    models, evaluations, theta, _ = \
        input_check_model(models, theta, None, N)
    noise_min = []
    noise_max = []
    for i in tqdm.trange(N):
        sample, rdm_idx, pattern_idx = \
            bootstrap_sample(data, rdm_descriptor=rdm_descriptor,
                             pattern_descriptor=pattern_descriptor)
        if len(np.unique(pattern_idx)) >= 3:
            for j, mod in enumerate(models):
                rdm_pred = mod.predict_rdm(theta=theta[j])
                rdm_pred = rdm_pred.subsample_pattern(pattern_descriptor,
                                                      pattern_idx)
                evaluations[i, j] = np.mean(compare(rdm_pred, sample,
                                                    method))
            if boot_noise_ceil:
                noise_min_sample, noise_max_sample = boot_noise_ceiling(
                    sample, method=method, rdm_descriptor=rdm_descriptor)
                noise_min.append(noise_min_sample)
                noise_max.append(noise_max_sample)
        else:
            evaluations[i, :] = np.nan
            noise_min.append(np.nan)
            noise_max.append(np.nan)
    if boot_noise_ceil:
        eval_ok = np.isfinite(evaluations[:, 0])
        noise_ceil = np.array([noise_min, noise_max])
        var = np.cov(np.concatenate([evaluations[eval_ok, :].T,
                                     noise_ceil[:, eval_ok]]))
        variances = var[:-2, :-2]
        noise_ceil_var = var[:, -2:]
    else:
        eval_ok = np.isfinite(evaluations[:, 0])
        noise_ceil = np.array(boot_noise_ceiling(
            data, method=method, rdm_descriptor=rdm_descriptor))
        variances = np.cov(evaluations[eval_ok, :].T)
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
        models(pyrsa.model.Model or list): models to be evaluated
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
    models, evaluations, theta, _ = \
        input_check_model(models, theta, None, N)
    noise_min = []
    noise_max = []
    for i in tqdm.trange(N):
        sample, pattern_idx = \
            bootstrap_sample_pattern(data, pattern_descriptor)
        if len(np.unique(pattern_idx)) >= 3:
            for j, mod in enumerate(models):
                rdm_pred = mod.predict_rdm(theta=theta[j])
                rdm_pred = rdm_pred.subsample_pattern(pattern_descriptor,
                                                      pattern_idx)
                evaluations[i, j] = np.mean(compare(rdm_pred, sample,
                                                    method))
            if boot_noise_ceil:
                noise_min_sample, noise_max_sample = boot_noise_ceiling(
                    sample, method=method, rdm_descriptor=rdm_descriptor)
                noise_min.append(noise_min_sample)
                noise_max.append(noise_max_sample)
        else:
            evaluations[i, :] = np.nan
            noise_min.append(np.nan)
            noise_max.append(np.nan)
    if boot_noise_ceil:
        eval_ok = np.isfinite(evaluations[:, 0])
        noise_ceil = np.array([noise_min, noise_max])
        var = np.cov(np.concatenate([evaluations[eval_ok, :].T,
                                     noise_ceil[:, eval_ok]]))
        variances = var[:-2, :-2]
        noise_ceil_var = var[:, -2:]
    else:
        eval_ok = np.isfinite(evaluations[:, 0])
        noise_ceil = np.array(boot_noise_ceiling(
            data, method=method, rdm_descriptor=rdm_descriptor))
        variances = np.cov(evaluations[eval_ok, :].T)
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
        models(pyrsa.model.Model or list of these): models to be evaluated
        data(pyrsa.rdm.RDMs): data to evaluate on
        theta(numpy.ndarray): parameter vector for the models
        method(string): comparison method to use
        N(int): number of samples
        rdm_descriptor(string): rdm_descriptor to group rdms for bootstrap

    Returns:
        numpy.ndarray: vector of evaluations

    """
    models, evaluations, theta, _ = input_check_model(models, theta, None, N)
    noise_min = []
    noise_max = []
    for i in tqdm.trange(N):
        sample, rdm_idx = bootstrap_sample_rdm(data, rdm_descriptor)
        for j, mod in enumerate(models):
            rdm_pred = mod.predict_rdm(theta=theta[j])
            evaluations[i, j] = np.mean(compare(rdm_pred, sample,
                                                method))
        if boot_noise_ceil:
            noise_min_sample, noise_max_sample = boot_noise_ceiling(
                sample, method=method, rdm_descriptor=rdm_descriptor)
            noise_min.append(noise_min_sample)
            noise_max.append(noise_max_sample)
    if boot_noise_ceil:
        eval_ok = np.isfinite(evaluations[:, 0])
        noise_ceil = np.array([noise_min, noise_max])
        var = np.cov(np.concatenate([evaluations[eval_ok, :].T,
                                     noise_ceil[:, eval_ok]]))
        variances = var[:-2, :-2]
        noise_ceil_var = var[:, -2:]
    else:
        eval_ok = np.isfinite(evaluations[:, 0])
        noise_ceil = np.array(boot_noise_ceiling(
            data, method=method, rdm_descriptor=rdm_descriptor))
        variances = np.cov(evaluations[eval_ok, :].T)
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
    if isinstance(models, Model):
        models = [models]
    evaluations = []
    noise_ceil = []
    for i in range(len(train_set)):
        train = train_set[i]
        test = test_set[i]
        if (train[0].n_rdm == 0 or test[0].n_rdm == 0 or
                train[0].n_cond <= 2 or test[0].n_cond <= 2):
            evals = np.empty(len(models)) * np.nan
        else:
            models, evals, _, fitter = \
                input_check_model(models, None, fitter)
            for j, model in enumerate(models):
                theta = fitter[j](model, train[0], method=method,
                                  pattern_idx=train[1],
                                  pattern_descriptor=pattern_descriptor)
                pred = model.predict_rdm(theta)
                pred = pred.subsample_pattern(by=pattern_descriptor,
                                              value=test[1])
                evals[j] = np.mean(compare(pred, test[0], method))
            if ceil_set is None and calc_noise_ceil:
                noise_ceil.append(boot_noise_ceiling(
                    rdms.subsample_pattern(by=pattern_descriptor,
                                           value=test[1]), method=method))
        evaluations.append(evals)
    evaluations = np.array(evaluations).T  # .T to switch models/set order
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
                       k_pattern=None, k_rdm=None, N=1000, n_cv=2,
                       pattern_descriptor='index', rdm_descriptor='index',
                       random=True, boot_type='both', use_correction=True,
                       calc_noise_ceil=True):
    """evaluates a set of models by k-fold crossvalidation within a bootstrap

    If a k is set to 1 no crossvalidation is performed over the
    corresponding dimension.

    As especially crossvalidation over patterns/conditions creates
    variance in the cv result for a single variance the default setting
    of n_cv=1 inflates the estimated variance. Setting this value
    higher will decrease this effect at the cost of more computation time.

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
        n_cv(int) : number of crossvalidation runs per sample (default: 1)
        pattern_descriptor(string): descriptor to group patterns
        rdm_descriptor(string): descriptor to group rdms
        random(bool): randomize group assignments (default: True)
        boot_type(String): which dimension to bootstrap over (default: 'both')
            alternatives: 'rdm', 'pattern'
        use_correction(bool): switch for the correction for the
            variance caused by crossvalidation (default: True)
        calc_noise_ceil(bool or String): how to calculate noise ceiling
            False, 'False', None: don't calculate noise_ceiling
            True, 'fix': default, calculate one noise ceiling from the
                whole sample
            'boot': calculate noise_ceiling for each bootstrapsample
                + covariance estimate with model evaluations

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
    if calc_noise_ceil == 'False':
        calc_noise_ceil = False
    evaluations = np.zeros((N, len(models), k_pattern * k_rdm, n_cv))
    noise_ceil = np.zeros((2, N, n_cv))
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
            for i_rep in range(n_cv):
                train_set, test_set, ceil_set = sets_k_fold(
                    sample,
                    pattern_descriptor=pattern_descriptor,
                    rdm_descriptor=rdm_descriptor,
                    k_pattern=k_pattern, k_rdm=k_rdm, random=random)
                if calc_noise_ceil == 'boot':
                    if k_rdm > 1 or k_pattern > 1:
                        cv_nc = cv_noise_ceiling(
                            sample, ceil_set, test_set,
                            method=method,
                            pattern_descriptor=pattern_descriptor)
                        noise_ceil[:, i_sample, i_rep] = cv_nc
                    else:
                        nc = boot_noise_ceiling(
                            sample,
                            method=method,
                            rdm_descriptor=rdm_descriptor)
                        noise_ceil[:, i_sample, i_rep] = nc
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
                evaluations[i_sample, :, :, i_rep] = cv_result.evaluations[0]
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
    eval_ok = ~np.any(np.any(np.any(np.isnan(evaluations),
                                    axis=-1), axis=-1), axis=-1)
    evals_nonan = np.mean(np.mean(evaluations[eval_ok], -1), -1)
    if use_correction and n_cv > 1:
        # we essentially project from the two points for 1 repetition and
        # for n_cv repetitions to infinitely many cv repetitions
        evals_1 = np.mean(evaluations[eval_ok], -2)
        var_mean = np.cov(evals_nonan.T)
        var_1 = []
        for i in range(n_cv):
            var_1.append(np.cov(evals_1[:, :, i].T))
        var_1 = np.mean(np.array(var_1), axis=0)
        # this is the main formula for the correction:
        variances = (n_cv * var_mean - var_1) / (n_cv - 1)
        if calc_noise_ceil == 'boot':
            # for the noise_ceiling we are interested in the covariance,
            # which should be correct from the mean estimates, as the covariance
            # of the crossvalidation noise should be 0
            noise_ceil_nonan = np.mean(noise_ceil[:, eval_ok], -1)
            vars_nc = np.cov(np.concatenate([evals_nonan.T, noise_ceil_nonan]))
            noise_ceil_var = vars_nc[:, -2:]
        else:
            noise_ceil_var = None
    else:
        if use_correction:
            raise Warning('correction requested, but only one cv run'
                          + ' per sample requested. This is invalid!'
                          + ' We do not use the correction for now.')
        if calc_noise_ceil == 'boot':
            noise_ceil_nonan = np.mean(noise_ceil[:, eval_ok], -1)
            variances = np.cov(np.concatenate([evals_nonan.T, noise_ceil_nonan]))
            noise_ceil_var = variances[:, -2:]
            variances = variances[:-2, :-2]
        else:
            variances = np.cov(evals_nonan.T)
            noise_ceil_var = None
    if calc_noise_ceil == 'boot':
        pass
    elif calc_noise_ceil:
        if k_rdm > 1 or k_pattern > 1:
            train_set, test_set, ceil_set = sets_k_fold(
                data,
                pattern_descriptor=pattern_descriptor,
                rdm_descriptor=rdm_descriptor,
                k_pattern=k_pattern, k_rdm=k_rdm, random=random)
            noise_ceil = cv_noise_ceiling(
                data, ceil_set, test_set,
                method=method,
                pattern_descriptor=pattern_descriptor)
        else:
            noise_ceil = boot_noise_ceiling(
                data,
                method=method,
                rdm_descriptor=rdm_descriptor)
    else:
        noise_ceil = None
        noise_ceil_var = None
    result = Result(models, evaluations, method=method,
                    cv_method=cv_method, noise_ceiling=noise_ceil,
                    variances=variances, dof=dof,
                    noise_ceil_var=noise_ceil_var)
    return result


def _concat_sampling(sample1, sample2):
    """ computes an index vector for the sequential sampling with sample1
    and sample2
    """
    sample_out = [[i_samp1 for i_samp1 in sample1 if i_samp1 == i_samp2]
                  for i_samp2 in sample2]
    return sum(sample_out, [])
