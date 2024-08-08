#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate model performance
"""

import numpy as np
import tqdm
from rsatoolbox.rdm import compare
from rsatoolbox.inference import bootstrap_sample
from rsatoolbox.inference import bootstrap_sample_rdm
from rsatoolbox.inference import bootstrap_sample_pattern
from rsatoolbox.model import Model
from rsatoolbox.util.inference_util import input_check_model
from rsatoolbox.util.inference_util import default_k_pattern, default_k_rdm
from .result import Result
from .crossvalsets import sets_k_fold, sets_random
from .noise_ceiling import boot_noise_ceiling
from .noise_ceiling import cv_noise_ceiling


def eval_dual_bootstrap(
        models, data, method='cosine', fitter=None,
        k_pattern=1, k_rdm=1, N=1000, n_cv=2,
        pattern_descriptor='index', rdm_descriptor='index',
        use_correction=True):
    """dual bootstrap evaluation of models
    i.e. models are evaluated in a bootstrap over rdms, one over patterns
    and a bootstrap over both using the same bootstrap samples for each.
    The variance estimates from these bootstraps are then combined into
    a better overall estimate for the variance.

    This method allows the incorporation of crossvalidation inside the
    bootstrap to handle fitted models.
    To activate this set k_rdm and k_pattern as described below.

    Crossvalidation creates variance in the results for a single bootstrap
    sample, because different assginments to the training and test group
    lead to different results. To correct for this, we apply a formula
    which estimates the variance we expect if we evaluated all possible
    crossvalidation assignments from n_cv different assignments per bootstrap
    sample.
    In our statistical evaluations we saw that many bootstrap samples and
    few different crossvalidation assignments are optimal to minimize the
    variance of the variance estimate. Thus, this function by default
    applies this correction formula and sets n_cv=2, i.e. performs only two
    different assignments per fold.
    This function nonetheless performs full crossvalidation schemes, i.e.
    in every bootstrap sample all crossvalidation folds are evaluated such
    that each RDM and each condition is in the test set n_cv times.

    The k_[] parameters control the cross-validation per sample. They give
    the number of crossvalidation folds to be created along this dimension.
    If a k is set to 1 no crossvalidation is performed over the
    corresponding dimension.
    by default ks are set by rsatoolbox.util.inference_util.default_k_pattern
    and rsatoolbox.util.inference_util.default_k_rdm based on the number of
    rdms and patterns provided. the ks are then in the range 2-5.

    Using the []_descriptor inputs you may make the crossvalidation and
    bootstrap aware of groups of rdms or conditions to be handled en block.
    Conditions with the same entry will be sampled in or out of the bootstrap
    together and will be assigned to cross-calidation folds together.

    models should be a list of models. data the RDMs object to evaluate against
    method the method for comparing the predictions and the data. fitter may
    provide a non-default funcion or list of functions to fit the models.

    Args:
        models(rsatoolbox.model.Model): models to be evaluated
        data(rsatoolbox.rdm.RDMs): RDM data to use
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
    if k_rdm == 1 and k_pattern == 1:
        n_cv = 1
        use_correction = False
    if isinstance(models, Model):
        models = [models]
    evaluations = np.zeros((N, len(models), k_pattern * k_rdm, n_cv, 3))
    noise_ceil = np.zeros((2, N, n_cv, 3))
    for i_sample in tqdm.trange(N):
        sample, rdm_idx, pattern_idx = bootstrap_sample(
            data,
            rdm_descriptor=rdm_descriptor,
            pattern_descriptor=pattern_descriptor)
        sample_rdm = data.subsample(rdm_descriptor, rdm_idx)
        sample_pattern = data.subsample_pattern(
            pattern_descriptor, pattern_idx)
        if len(np.unique(rdm_idx)) >= k_rdm \
           and len(np.unique(pattern_idx)) >= 3 * k_pattern:
            for i_rep in range(n_cv):
                evals, cv_nc = _internal_cv(
                    models, sample,
                    pattern_descriptor, rdm_descriptor, pattern_idx,
                    k_pattern, k_rdm,
                    method, fitter)
                noise_ceil[:, i_sample, i_rep, 0] = cv_nc
                evaluations[i_sample, :, :, i_rep, 0] = evals[0]
                evals, cv_nc = _internal_cv(
                    models, sample_rdm,
                    pattern_descriptor, rdm_descriptor,
                    np.unique(data.pattern_descriptors[pattern_descriptor]),
                    k_pattern, k_rdm,
                    method, fitter)
                noise_ceil[:, i_sample, i_rep, 1] = cv_nc
                evaluations[i_sample, :, :, i_rep, 1] = evals[0]
                evals, cv_nc = _internal_cv(
                    models, sample_pattern,
                    pattern_descriptor, rdm_descriptor, pattern_idx,
                    k_pattern, k_rdm,
                    method, fitter)
                noise_ceil[:, i_sample, i_rep, 2] = cv_nc
                evaluations[i_sample, :, :, i_rep, 2] = evals[0]
        else:  # sample does not allow desired crossvalidation
            evaluations[i_sample, :, :, :, :] = np.nan
            noise_ceil[:, i_sample, :, :] = np.nan
    cv_method = 'dual_bootstrap'
    dof = min(data.n_rdm, data.n_cond) - 1
    eval_ok = ~np.isnan(evaluations[:, 0, 0, 0, 0])
    if use_correction and n_cv > 1:
        # we essentially project from the two points for 1 repetition and
        # for n_cv repetitions to infinitely many cv repetitions
        evals_nonan = np.mean(np.mean(evaluations[eval_ok], -2), -2)
        evals_1 = np.mean(evaluations[eval_ok], -3)
        noise_ceil_nonan = np.mean(
            noise_ceil[:, eval_ok], -2).transpose([1, 0, 2])
        noise_ceil_1 = noise_ceil[:, eval_ok].transpose([1, 0, 2, 3])
        matrix = np.concatenate([evals_nonan, noise_ceil_nonan], 1)
        matrix -= np.mean(matrix, 0, keepdims=True)
        var_mean = np.einsum('ijk,ilk->kjl', matrix, matrix) \
            / (matrix.shape[0] - 1)
        matrix_1 = np.concatenate([evals_1, noise_ceil_1], 1)
        matrix_1 -= np.mean(matrix_1, 0, keepdims=True)
        var_1 = np.einsum('ijmk,ilmk->kjl', matrix_1, matrix_1) \
            / (matrix_1.shape[0] - 1) / matrix_1.shape[2]
        # this is the main formula for the correction:
        variances = (n_cv * var_mean - var_1) / (n_cv - 1)
    else:
        if use_correction:
            raise Warning('correction requested, but only one cv run'
                          + ' per sample requested. This is invalid!'
                          + ' We do not use the correction for now.')
        evals_nonan = np.mean(np.mean(evaluations[eval_ok], -2), -2)
        noise_ceil_nonan = np.mean(
            noise_ceil[:, eval_ok], -2).transpose([1, 0, 2])
        matrix = np.concatenate([evals_nonan, noise_ceil_nonan], 1)
        matrix -= np.mean(matrix, 0, keepdims=True)
        variances = np.einsum('ijk,ilk->kjl', matrix, matrix) \
            / (matrix.shape[0] - 1)
    result = Result(models, evaluations, method=method,
                    cv_method=cv_method, noise_ceiling=noise_ceil,
                    variances=variances, dof=dof, n_rdm=data.n_rdm,
                    n_pattern=data.n_cond)
    return result


def eval_fixed(models, data, theta=None, method='cosine'):
    """evaluates models on data, without any bootstrapping or
    cross-validation

    Args:
        models(list of rsatoolbox.model.Model or list): models to be evaluated
        data(rsatoolbox.rdm.RDMs): data to evaluate on
        theta(numpy.ndarray): parameter vector for the models
        method(string): comparison method to use

    Returns:
        float: evaluation

    """
    models, evaluations, theta, _ = input_check_model(models, theta, None, 1)
    evaluations = np.repeat(np.expand_dims(evaluations, -1),
                            data.n_rdm, -1)
    for k, model in enumerate(models):
        rdm_pred = model.predict_rdm(theta=theta[k])
        evaluations[k] = compare(rdm_pred, data, method)
    evaluations = evaluations.reshape((1, len(models), data.n_rdm))
    noise_ceil = boot_noise_ceiling(
        data, method=method, rdm_descriptor='index')
    if data.n_rdm > 1:
        variances = np.cov(evaluations[0], ddof=0) \
            / evaluations.shape[-1]
        dof = evaluations.shape[-1] - 1
    else:
        variances = None
        dof = 0
    result = Result(models, evaluations, method=method,
                    cv_method='fixed', noise_ceiling=noise_ceil,
                    variances=variances, dof=dof, n_rdm=data.n_rdm,
                    n_pattern=None)
    result.n_pattern = data.n_cond
    return result


def eval_bootstrap(models, data, theta=None, method='cosine', N=1000,
                   pattern_descriptor='index', rdm_descriptor='index',
                   boot_noise_ceil=True):
    """evaluates models on data
    performs bootstrapping to get a sampling distribution

    Args:
        models(rsatoolbox.model.Model or list): models to be evaluated
        data(rsatoolbox.rdm.RDMs): data to evaluate on
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
        sample, _, pattern_idx = \
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
        variances = np.cov(np.concatenate([evaluations[eval_ok, :].T,
                                           noise_ceil[:, eval_ok]]))
    else:
        eval_ok = np.isfinite(evaluations[:, 0])
        noise_ceil = np.array(boot_noise_ceiling(
            data, method=method, rdm_descriptor=rdm_descriptor))
        variances = np.cov(evaluations[eval_ok, :].T)
    dof = min(data.n_rdm, data.n_cond) - 1
    result = Result(models, evaluations, method=method,
                    cv_method='bootstrap', noise_ceiling=noise_ceil,
                    variances=variances, dof=dof, n_rdm=data.n_rdm,
                    n_pattern=data.n_cond)
    return result


def eval_bootstrap_pattern(models, data, theta=None, method='cosine', N=1000,
                           pattern_descriptor='index', rdm_descriptor='index',
                           boot_noise_ceil=True):
    """evaluates a models on data
    performs bootstrapping over patterns to get a sampling distribution

    Args:
        models(rsatoolbox.model.Model or list): models to be evaluated
        data(rsatoolbox.rdm.RDMs): data to evaluate on
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
        variances = np.cov(np.concatenate([evaluations[eval_ok, :].T,
                                           noise_ceil[:, eval_ok]]))
    else:
        eval_ok = np.isfinite(evaluations[:, 0])
        noise_ceil = np.array(boot_noise_ceiling(
            data, method=method, rdm_descriptor=rdm_descriptor))
        variances = np.cov(evaluations[eval_ok, :].T)
    dof = data.n_cond - 1
    result = Result(models, evaluations, method=method,
                    cv_method='bootstrap_pattern', noise_ceiling=noise_ceil,
                    variances=variances, dof=dof, n_rdm=None,
                    n_pattern=data.n_cond)
    result.n_rdm = data.n_rdm
    return result


def eval_bootstrap_rdm(models, data, theta=None, method='cosine', N=1000,
                       rdm_descriptor='index', boot_noise_ceil=True):
    """evaluates models on data
    performs bootstrapping to get a sampling distribution

    Args:
        models(rsatoolbox.model.Model or list of these): models to be evaluated
        data(rsatoolbox.rdm.RDMs): data to evaluate on
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
        sample, _ = bootstrap_sample_rdm(data, rdm_descriptor)
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
        variances = np.cov(np.concatenate([evaluations[eval_ok, :].T,
                                           noise_ceil[:, eval_ok]]))
    else:
        eval_ok = np.isfinite(evaluations[:, 0])
        noise_ceil = np.array(boot_noise_ceiling(
            data, method=method, rdm_descriptor=rdm_descriptor))
        variances = np.cov(evaluations[eval_ok, :].T)
    dof = data.n_rdm - 1
    variances = np.cov(evaluations.T)
    result = Result(models, evaluations, method=method,
                    cv_method='bootstrap_rdm', noise_ceiling=noise_ceil,
                    variances=variances, dof=dof, n_rdm=data.n_rdm,
                    n_pattern=None)
    result.n_pattern = data.n_cond
    return result


def crossval(models, rdms, train_set, test_set, ceil_set=None, method='cosine',
             fitter=None, pattern_descriptor='index', calc_noise_ceil=True):
    """evaluates models on cross-validation sets

    Args:
        models(rsatoolbox.model.Model): models to be evaluated
        rdms(rsatoolbox.rdm.RDMs): full dataset
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
    for i, train in enumerate(train_set):
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
                                           value=test[1]),
                    method=method))
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
                       boot_type='both', use_correction=True):
    """evaluates a set of models by k-fold crossvalidation within a bootstrap

    Crossvalidation creates variance in the results for a single bootstrap
    sample, because different assginments to the training and test group
    lead to different results. To correct for this, we apply a formula
    which estimates the variance we expect if we evaluated all possible
    crossvalidation assignments from n_cv different assignments per bootstrap
    sample.
    In our statistical evaluations we saw that many bootstrap samples and
    few different crossvalidation assignments are optimal to minimize the
    variance of the variance estimate. Thus, this function by default
    applies this correction formula and sets n_cv=2, i.e. performs only two
    different assignments per fold.
    This function nonetheless performs full crossvalidation schemes, i.e.
    in every bootstrap sample all crossvalidation folds are evaluated such
    that each RDM and each condition is in the test set n_cv times. For the
    even more optimized version which computes only two randomly chosen test
    sets see bootstrap_cv_random.

    The k_[] parameters control the cross-validation per sample. They give
    the number of crossvalidation folds to be created along this dimension.
    If a k is set to 1 no crossvalidation is performed over the
    corresponding dimension.
    by default ks are set by rsatoolbox.util.inference_util.default_k_pattern
    and rsatoolbox.util.inference_util.default_k_rdm based on the number of
    rdms and patterns provided. the ks are then in the range 2-5.

    Using the []_descriptor inputs you may make the crossvalidation and
    bootstrap aware of groups of rdms or conditions to be handled en block.
    Conditions with the same entry will be sampled in or out of the bootstrap
    together and will be assigned to cross-calidation folds together.

    Using the boot_type argument you may choose the dimension to bootstrap.
    By default both conditions and RDMs are resampled. You may alternatively
    choose to resample only one of them by passing 'rdm' or 'pattern'.

    models should be a list of models. data the RDMs object to evaluate against
    method the method for comparing the predictions and the data. fitter may
    provide a non-default funcion or list of functions to fit the models.

    Args:
        models(rsatoolbox.model.Model): models to be evaluated
        data(rsatoolbox.rdm.RDMs): RDM data to use
        method(string): comparison method to use
        fitter(function): fitting method for models
        k_pattern(int): #folds over patterns
        k_rdm(int): #folds over rdms
        N(int): number of bootstrap samples (default: 1000)
        n_cv(int) : number of crossvalidation runs per sample (default: 1)
        pattern_descriptor(string): descriptor to group patterns
        rdm_descriptor(string): descriptor to group rdms
        boot_type(String): which dimension to bootstrap over (default: 'both')
            alternatives: 'rdm', 'pattern'
        use_correction(bool): switch for the correction for the
            variance caused by crossvalidation (default: True)

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
        if n_rdm == 1:
            k_rdm = 1
        else:
            k_rdm = default_k_rdm((1 - 1 / np.exp(1)) * n_rdm)
    if isinstance(models, Model):
        models = [models]
    evaluations = np.empty((N, len(models), k_pattern * k_rdm, n_cv))
    noise_ceil = np.empty((2, N, n_cv))
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
                evals, cv_nc = _internal_cv(
                    models, sample,
                    pattern_descriptor, rdm_descriptor, pattern_idx,
                    k_pattern, k_rdm,
                    method, fitter)
                noise_ceil[:, i_sample, i_rep] = cv_nc
                evaluations[i_sample, :, :, i_rep] = evals[0]
        else:  # sample does not allow desired crossvalidation
            evaluations[i_sample, :, :] = np.nan
            noise_ceil[:, i_sample] = np.nan
    if boot_type == 'both':
        cv_method = 'bootstrap_crossval'
        dof = min(data.n_rdm, data.n_cond) - 1
        n_rdm = data.n_rdm
        n_cond = data.n_cond
    elif boot_type == 'pattern':
        cv_method = 'bootstrap_crossval_pattern'
        dof = data.n_cond - 1
        n_rdm = None
        n_cond = data.n_cond
    elif boot_type == 'rdm':
        cv_method = 'bootstrap_crossval_rdm'
        dof = data.n_rdm - 1
        n_rdm = data.n_rdm
        n_cond = None
    eval_ok = ~np.isnan(evaluations[:, 0, 0, 0])
    if use_correction and n_cv > 1:
        # we essentially project from the two points for 1 repetition and
        # for n_cv repetitions to infinitely many cv repetitions
        evals_mean = np.mean(np.mean(evaluations[eval_ok], -1), -1)
        evals_1 = np.mean(evaluations[eval_ok], -2)
        noise_ceil_mean = np.mean(noise_ceil[:, eval_ok], -1)
        noise_ceil_1 = noise_ceil[:, eval_ok]
        var_mean = np.cov(
            np.concatenate([evals_mean.T, noise_ceil_mean]))
        var_1 = []
        for i in range(n_cv):
            var_1.append(np.cov(np.concatenate([
                evals_1[:, :, i].T, noise_ceil_1[:, :, i]])))
        var_1 = np.mean(np.array(var_1), axis=0)
        # this is the main formula for the correction:
        variances = (n_cv * var_mean - var_1) / (n_cv - 1)
    else:
        if use_correction:
            raise Warning('correction requested, but only one cv run'
                          + ' per sample requested. This is invalid!'
                          + ' We do not use the correction for now.')
        evals_nonan = np.mean(np.mean(evaluations[eval_ok], -1), -1)
        noise_ceil_nonan = np.mean(noise_ceil[:, eval_ok], -1)
        variances = np.cov(np.concatenate([evals_nonan.T, noise_ceil_nonan]))
    result = Result(models, evaluations, method=method,
                    cv_method=cv_method, noise_ceiling=noise_ceil,
                    variances=variances, dof=dof, n_rdm=n_rdm,
                    n_pattern=n_cond)
    return result


def eval_dual_bootstrap_random(
        models, data, method='cosine', fitter=None,
        n_pattern=None, n_rdm=None, N=1000, n_cv=2,
        pattern_descriptor='index', rdm_descriptor='index',
        boot_type='both', use_correction=True):
    """evaluates a set of models by a evaluating a few random crossvalidation
    folds per bootstrap.

    If a k is set to 1 no crossvalidation is performed over the
    corresponding dimension.

    As especially crossvalidation over patterns/conditions creates
    variance in the cv result for a single variance the default setting
    of n_cv=1 inflates the estimated variance. Setting this value
    higher will decrease this effect at the cost of more computation time.

    by default ks are set by rsatoolbox.util.inference_util.default_k_pattern
    and rsatoolbox.util.inference_util.default_k_rdm based on the number of
    rdms and patterns provided. the ks are then in the range 2-5.

    Args:
        models(rsatoolbox.model.Model): models to be evaluated
        data(rsatoolbox.rdm.RDMs): RDM data to use
        method(string): comparison method to use
        fitter(function): fitting method for models
        k_pattern(int): #folds over patterns
        k_rdm(int): #folds over rdms
        N(int): number of bootstrap samples (default: 1000)
        n_cv(int) : number of crossvalidation runs per sample (default: 1)
        pattern_descriptor(string): descriptor to group patterns
        rdm_descriptor(string): descriptor to group rdms
        boot_type(String): which dimension to bootstrap over (default: 'both')
            alternatives: 'rdm', 'pattern'
        use_correction(bool): switch for the correction for the
            variance caused by crossvalidation (default: True)

    Returns:
        numpy.ndarray: matrix of evaluations (N x k)

    """
    if n_pattern is None:
        n_pattern_all = len(np.unique(data.pattern_descriptors[
            pattern_descriptor]))
        k_pattern = default_k_pattern((1 - 1 / np.exp(1)) * n_pattern_all)
        n_pattern = int(np.floor(n_pattern_all / k_pattern))
    if n_rdm is None:
        n_rdm_all = len(np.unique(data.rdm_descriptors[
            rdm_descriptor]))
        k_rdm = default_k_rdm((1 - 1 / np.exp(1)) * n_rdm_all)
        n_rdm = int(np.floor(n_rdm_all / k_rdm))
    if isinstance(models, Model):
        models = [models]
    evaluations = np.zeros((N, len(models), n_cv))
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
        if len(np.unique(rdm_idx)) > n_rdm \
           and len(np.unique(pattern_idx)) >= 3 + n_pattern:
            train_set, test_set, ceil_set = sets_random(
                sample,
                pattern_descriptor=pattern_descriptor,
                rdm_descriptor=rdm_descriptor,
                n_pattern=n_pattern, n_rdm=n_rdm, n_cv=n_cv)
            if n_rdm > 0 or n_pattern > 0:
                nc = cv_noise_ceiling(
                    sample, ceil_set, test_set,
                    method=method,
                    pattern_descriptor=pattern_descriptor)
            else:
                nc = boot_noise_ceiling(
                    sample,
                    method=method,
                    rdm_descriptor=rdm_descriptor)
            noise_ceil[:, i_sample] = nc
            for test_s in test_set:
                test_s[1] = _concat_sampling(pattern_idx, test_s[1])
            for train_s in train_set:
                train_s[1] = _concat_sampling(pattern_idx, train_s[1])
            cv_result = crossval(
                models, sample,
                train_set, test_set,
                method=method, fitter=fitter,
                pattern_descriptor=pattern_descriptor,
                calc_noise_ceil=False)
            evaluations[i_sample, :, :] = cv_result.evaluations[0]
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
    if use_correction and n_cv > 1:
        # we essentially project from the two points for 1 repetition and
        # for n_cv repetitions to infinitely many cv repetitions
        evals_mean = np.mean(evaluations[eval_ok], -1)
        evals_1 = evaluations[eval_ok]
        noise_ceil_mean = np.mean(noise_ceil[:, eval_ok], -1)
        noise_ceil_1 = noise_ceil[:, eval_ok]
        var_mean = np.cov(
            np.concatenate([evals_mean.T, noise_ceil_mean]))
        var_1 = []
        for i in range(n_cv):
            var_1.append(np.cov(np.concatenate([
                evals_1[:, :, i].T, noise_ceil_1[:, :, i]])))
        var_1 = np.mean(np.array(var_1), axis=0)
        # this is the main formula for the correction:
        variances = (n_cv * var_mean - var_1) / (n_cv - 1)
    else:
        if use_correction:
            raise Warning('correction requested, but only one cv run'
                          + ' per sample requested. This is invalid!'
                          + ' We do not use the correction for now.')
        evals_nonan = np.mean(np.mean(evaluations[eval_ok], -1), -1)
        noise_ceil_nonan = np.mean(noise_ceil[:, eval_ok], -1)
        variances = np.cov(np.concatenate([evals_nonan.T, noise_ceil_nonan]))
    result = Result(models, evaluations, method=method,
                    cv_method=cv_method, noise_ceiling=noise_ceil,
                    variances=variances, dof=dof, n_rdm=data.n_rdm,
                    n_pattern=data.n_cond)
    return result


def _concat_sampling(sample1, sample2):
    """ computes an index vector for the sequential sampling with sample1
    and sample2
    """
    sample_out = [[i_samp1 for i_samp1 in sample1 if i_samp1 == i_samp2]
                  for i_samp2 in sample2]
    return sum(sample_out, [])


def _internal_cv(models, sample,
                 pattern_descriptor, rdm_descriptor, pattern_idx,
                 k_pattern, k_rdm,
                 method, fitter):
    """ runs a crossvalidation for use in bootstrap"""
    train_set, test_set, ceil_set = sets_k_fold(
        sample,
        pattern_descriptor=pattern_descriptor,
        rdm_descriptor=rdm_descriptor,
        k_pattern=k_pattern, k_rdm=k_rdm, random=True)
    if k_rdm > 1 or k_pattern > 1:
        nc = cv_noise_ceiling(
            sample, ceil_set, test_set,
            method=method,
            pattern_descriptor=pattern_descriptor)
    else:
        nc = boot_noise_ceiling(
            sample,
            method=method,
            rdm_descriptor=rdm_descriptor)
    for test_s in test_set:
        test_s[1] = _concat_sampling(pattern_idx, test_s[1])
    for train_s in train_set:
        train_s[1] = _concat_sampling(pattern_idx, train_s[1])
    cv_result = crossval(
        models, sample,
        train_set, test_set,
        method=method, fitter=fitter,
        pattern_descriptor=pattern_descriptor,
        calc_noise_ceil=False)
    return cv_result.evaluations, nc
