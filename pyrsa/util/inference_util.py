#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inference module utilities
"""

import numpy as np
import scipy.stats as stats
from scipy.stats import rankdata, wilcoxon
from pyrsa.model import Model
from pyrsa.rdm import RDMs
from .matrix import pairwise_contrast
from .rdm_utils import batch_to_matrices
from collections.abc import Iterable


def input_check_model(models, theta=None, fitter=None, N=1):
    """ Checks whether model related inputs to evaluations are valid and
    generates an evaluation-matrix of fitting size.

    Args:
        model : [list of] pyrsa.rdm.RDMs
            the models to be evaluated
        theta : numpy.ndarray or list , optional
            Parameter(s) for the model(s). The default is None.
        fitter : [list of] function, optional
            fitting function to overwrite the model default.
            The default is None, i.e. keep default
        N : int, optional
            number of samples/rows in evaluations matrix. The default is 1.

    Returns:
        evaluations : numpy.ndarray
            empty evaluations-matrix
        theta : list
            the processed and checked model parameters
        fitter : [list of] functions
            checked and processed fitter functions

    """
    if isinstance(models, Model):
        models = [models]
    elif not isinstance(models, Iterable):
        raise ValueError('model should be a pyrsa.model.Model or a list of'
                         + ' such objects')
    if N > 1:
        evaluations = np.zeros((N, len(models)))
    else:
        evaluations = np.zeros(len(models))
    if theta is not None:
        assert isinstance(theta, Iterable), 'If a list of models is' \
            + ' passed theta must be a list of parameters'
        assert len(models) == len(theta), 'there should equally many' \
            + ' models as parameters'
    else:
        theta = [None] * len(models)
    if fitter is None:
        fitter = [None] * len(models)
    elif isinstance(fitter, Iterable):
        assert len(fitter) == len(models), 'if fitters are passed ' \
            + 'there should be as many as models'
    else:
        fitter = [fitter] * len(models)
    for k, model in enumerate(models):
        if fitter[k] is None:
            fitter[k] = model.default_fitter
    return models, evaluations, theta, fitter


def pool_rdm(rdms, method='cosine'):
    """pools multiple RDMs into the one with maximal performance under a given
    evaluation metric
    rdm_descriptors of the generated rdms are empty

    Args:
        rdms (pyrsa.rdm.RDMs):
            RDMs to be pooled
        method : String, optional
            Which comparison method to optimize for. The default is 'cosine'.

    Returns:
        pyrsa.rdm.RDMs: the pooled RDM, i.e. a RDM with maximal performance
            under the chosen method

    """
    rdm_vec = rdms.get_vectors()
    if method == 'euclid':
        rdm_vec = _nan_mean(rdm_vec)
    elif method == 'cosine':
        rdm_vec = rdm_vec / np.sqrt(np.nanmean(rdm_vec ** 2, axis=1,
                                               keepdims=True))
        rdm_vec = _nan_mean(rdm_vec)
    elif method == 'corr':
        rdm_vec = rdm_vec - np.nanmean(rdm_vec, axis=1, keepdims=True)
        rdm_vec = rdm_vec / np.nanstd(rdm_vec, axis=1, keepdims=True)
        rdm_vec = _nan_mean(rdm_vec)
        rdm_vec = rdm_vec - np.nanmin(rdm_vec)
    elif method == 'cosine_cov':
        rdm_vec = rdm_vec / np.sqrt(np.nanmean(rdm_vec ** 2, axis=1,
                                               keepdims=True))
        rdm_vec = _nan_mean(rdm_vec)
    elif method == 'corr_cov':
        rdm_vec = rdm_vec - np.nanmean(rdm_vec, axis=1, keepdims=True)
        rdm_vec = rdm_vec / np.nanstd(rdm_vec, axis=1, keepdims=True)
        rdm_vec = _nan_mean(rdm_vec)
        rdm_vec = rdm_vec - np.nanmin(rdm_vec)
    elif method == 'spearman' or method == 'rho-a':
        rdm_vec = np.array([_nan_rank_data(v) for v in rdm_vec])
        rdm_vec = _nan_mean(rdm_vec)
    elif method == 'rho-a':
        rdm_vec = np.array([_nan_rank_data(v) for v in rdm_vec])
        rdm_vec = _nan_mean(rdm_vec)
    elif method == 'kendall' or method == 'tau-b':
        Warning('Noise ceiling for tau based on averaged ranks!')
        rdm_vec = np.array([_nan_rank_data(v) for v in rdm_vec])
        rdm_vec = _nan_mean(rdm_vec)
    elif method == 'tau-a':
        Warning('Noise ceiling for tau based on averaged ranks!')
        rdm_vec = np.array([_nan_rank_data(v) for v in rdm_vec])
        rdm_vec = _nan_mean(rdm_vec)
    else:
        raise ValueError('Unknown RDM comparison method requested!')
    return RDMs(rdm_vec,
                dissimilarity_measure=rdms.dissimilarity_measure,
                descriptors=rdms.descriptors,
                rdm_descriptors=None,
                pattern_descriptors=rdms.pattern_descriptors)


def _nan_mean(rdm_vector):
    """ takes the average over a rdm_vector with nans for masked entries
    without a warning

    Args:
        rdm_vector(numpy.ndarray): set of rdm_vectors to be averaged

    Returns:
        rdm_mean(numpy.ndarray): the mean rdm

    """
    nan_idx = ~np.isnan(rdm_vector[0])
    mean_values = np.mean(rdm_vector[:, nan_idx], axis=0)
    rdm_mean = np.empty((1, rdm_vector.shape[1])) * np.nan
    rdm_mean[:, nan_idx] = mean_values
    return rdm_mean


def _nan_rank_data(rdm_vector):
    """ rank_data for vectors with nan entries

    Args:
        rdm_vector(numpy.ndarray): the vector to be rank_transformed

    Returns:
        ranks(numpy.ndarray): the ranks with nans where the original vector
            had nans

    """
    ranks_no_nan = rankdata(rdm_vector[~np.isnan(rdm_vector)])
    ranks = np.ones_like(rdm_vector) * np.nan
    ranks[~np.isnan(rdm_vector)] = ranks_no_nan
    return ranks


def all_tests(evaluations, noise_ceil, test_type='t-test',
              variances=None, noise_ceil_var=None, dof=1):
    """wrapper running all tests necessary for the model plot
    -> pairwise tests, tests against 0 and against noise ceiling


    Args:
        evaluations (numpy.ndarray):
            model evaluations to be compared
            (should be 3D: bootstrap x models x subjects or repeats)
        noise_ceil (numpy.ndarray):
            noise_ceiling estimate(s) to compare against
        test_type(Strinng):
            't-test' : t-test bases tests using variances
            'bootstrap' : Direct bootstrap sample based tests
            'ranksum' : Wilcoxon signed rank-sum tests

    Returns:
        numpy.ndarrays: p_pairwise, p_zero, p_noise

    """
    if test_type == 't-test':
        p_pairwise = t_tests(evaluations, variances, dof=dof)
        p_zero = t_test_0(evaluations, variances, dof=dof)
        p_noise = t_test_nc(evaluations, variances, np.mean(noise_ceil[0]),
                            noise_ceil_var, dof)
    elif test_type == 'bootstrap':
        if len(noise_ceil.shape) > 1:
            noise_lower_bs = noise_ceil[0]
            noise_lower_bs.shape = (noise_ceil.shape[0], 1)
        else:
            noise_lower_bs = noise_ceil[0].reshape(1, 1)
        p_pairwise = pair_tests(evaluations)
        p_zero = ((evaluations < 0).sum(axis=0) + 1) / evaluations.shape[0]
        diffs = noise_lower_bs - evaluations
        p_noise = ((diffs < 0).sum(axis=0) + 1) / evaluations.shape[0]
    elif test_type == 'ranksum':
        noise_c = np.mean(noise_ceil[0])
        p_pairwise = ranksum_pair_test(evaluations)
        p_zero = ranksum_value_test(evaluations, 0)
        p_noise = ranksum_value_test(evaluations, noise_c)
    else:
        raise ValueError('test_type not recognized.\n'
                         + 'Options are: t-test, bootstrap, ranksum')
    return p_pairwise, p_zero, p_noise


def ranksum_pair_test(evaluations):
    """pairwise tests between models using the wilcoxon signed rank test


    Args:
        evaluations (numpy.ndarray):
            model evaluations to be compared
            (should be 3D: bootstrap x models x subjects or repeats)

    Returns:
        numpy.ndarray: matrix of proportions of opposit conclusions, i.e.
            p-values for the test

    """
    # check that the dimensionality is correct
    assert evaluations.ndim == 3, \
        'provided evaluations array has wrong dimensionality'
    n_model = evaluations.shape[1]
    # ignore bootstraps
    evaluations = np.nanmean(evaluations, 0)
    pvalues = np.empty((n_model, n_model))
    for i_model in range(n_model - 1):
        for j_model in range(i_model + 1, n_model):
            pvalues[i_model, j_model] = wilcoxon(
                evaluations[i_model], evaluations[j_model]).pvalue
            pvalues[j_model, i_model] = pvalues[i_model, j_model]
    np.fill_diagonal(pvalues, 1)
    return pvalues


def ranksum_value_test(evaluations, comp_value=0):
    """nonparametric wilcoxon signed rank test against a fixed value


    Args:
        evaluations (numpy.ndarray):
            model evaluations to be compared
            (should be 3D: bootstrap x models x subjects or repeats)
        comp_value(float):
            value to compare against

    Returns:
        float: p-value

    """
    # check that the dimensionality is correct
    assert evaluations.ndim == 3, \
        'provided evaluations array has wrong dimensionality'
    n_model = evaluations.shape[1]
    # ignore bootstraps
    evaluations = np.nanmean(evaluations, 0)
    pvalues = np.empty(n_model)
    for i_model in range(n_model):
        pvalues[i_model] = wilcoxon(
            evaluations[i_model] - comp_value).pvalue
    return pvalues


def pair_tests(evaluations):
    """pairwise bootstrapping significance tests for a difference in model
    performance.
    Tests add 1/len(evaluations) to each p-value and are computed as
    two sided tests, i.e. as 2 * the smaller proportion

    Args:
        evaluations (numpy.ndarray):
            model evaluations to be compared

    Returns:
        numpy.ndarray: matrix of proportions of opposit conclusions, i.e.
            p-values for the bootstrap test
    """
    proportions = np.zeros((evaluations.shape[1], evaluations.shape[1]))
    while len(evaluations.shape) > 2:
        evaluations = np.mean(evaluations, axis=-1)
    for i_model in range(evaluations.shape[1]-1):
        for j_model in range(i_model + 1, evaluations.shape[1]):
            proportions[i_model, j_model] = np.sum(
                evaluations[:, i_model] < evaluations[:, j_model]) \
                / (evaluations.shape[0] -
                   np.sum(evaluations[:, i_model] == evaluations[:, j_model]))
            proportions[j_model, i_model] = proportions[i_model, j_model]
    proportions = np.minimum(proportions, 1 - proportions) * 2
    proportions = (len(evaluations) - 1) / len(evaluations) * proportions \
        + 1 / len(evaluations)
    np.fill_diagonal(proportions, 1)
    return proportions


def t_tests(evaluations, variances, dof=1):
    """pairwise t_test based significant tests for a difference in model
    performance

    Take special care here preparing variances! This should be the covariance
    matrix for the model evaluations.

    Args:
        evaluations (numpy.ndarray):
            model evaluations to be tested, typically from a results object
        variances (numpy.ndarray):
            vector of model evaluation variances
            or covariance matrix of the model evaluations
        dof (integer):
            degrees of freedom used for the test (default=1)
            this input is overwritten if no variances are passed

    Returns:
        numpy.ndarray: matrix of proportions of opposit conclusions, i.e.
        p-values for the bootstrap test

    """
    if variances is None:
        raise ValueError('No variance estimates provided for t_test!')
    n_model = evaluations.shape[1]
    evaluations = np.mean(evaluations, 0)
    if len(variances.shape) == 1:
        variances = np.diag(variances)
    while evaluations.ndim > 1:
        evaluations = np.mean(evaluations, axis=-1)
    C = pairwise_contrast(np.arange(n_model))
    diffs = C @ evaluations
    var = np.diag(C @ variances @ C.T)
    t = diffs / np.sqrt(var)
    t = batch_to_matrices(np.array([t]))[0][0]
    p = 2 * (1 - stats.t.cdf(np.abs(t), dof))
    return p


def t_test_0(evaluations, variances, dof=1):
    """
    t-tests against 0 performance.

    Args:
        evaluations (numpy.ndarray):
            model evaluations to be tested, typically from a results object
        variances (numpy.ndarray):
            vector of model evaluation variances
            or covariance matrix of the model evaluations
        dof (integer):
            degrees of freedom used for the test (default=1)
            this input is overwritten if no variances are passed

    Returns:
        numpy.ndarray: p-values for the raw t-test of each model against 0.

    """
    if variances is None:
        raise ValueError('No variance estimates provided for t_test!')
    elif variances.ndim <= 1:
        variances = variances.reshape(1, 1)
    evaluations = np.mean(evaluations, 0)
    if len(variances.shape) == 1:
        variances = np.diag(variances)
    while evaluations.ndim > 1:
        evaluations = np.mean(evaluations, axis=-1)
    var = np.diag(variances)
    t = evaluations / np.sqrt(var)
    p = 1 - stats.t.cdf(t, dof)
    return p


def t_test_nc(evaluations, variances, noise_ceil, noise_ceil_var=None, dof=1):
    """
    t-tests against lower noise_ceiling.
    Technically this can be used to test evaluations against any fixed
    number.

    If noise_ceil_var is the covariance matrix of the model evaluations
    and the noise ceilings a normal t-test is performed.
    If noise_ceil_var is a single number or vector an indpendent t-test is
    performed
    If noise_ceil_var is None the noise_ceiling is treated as a fixed number

    Args:
        evaluations (numpy.ndarray):
            model evaluations to be tested, typically from a results object
        variances (numpy.ndarray):
            vector of model evaluation variances
            or covariance matrix of the model evaluations
            defaults to taking the variance over the third dimension
            of evaluations and setting dof based on the length of this
            dimension.
        noise_ceil (float):
            the average noise ceiling to test against.
        noise_ceil_var (numpy.ndarray):
            variance or covariance of the noise ceiling
        dof (integer):
            degrees of freedom used for the test (default=1)
            this input is overwritten if no variances are passed

    Returns:
        numpy.ndarray: p-values for the raw t-test of each model against
        the noise ceiling.

    """
    if variances is None:
        raise ValueError('No variance estimates provided for t_test!')
    elif variances.ndim <= 1:
        variances = variances.reshape(1, 1)
    if noise_ceil_var is not None:
        noise_ceil_var = np.array(noise_ceil_var)
        while noise_ceil_var.ndim > 1:
            noise_ceil_var = noise_ceil_var[:, 0]
    evaluations = np.nanmean(evaluations, 0)
    if len(variances.shape) == 1:
        variances = np.diag(variances)
    while evaluations.ndim > 1:
        evaluations = np.mean(evaluations, axis=-1)
    var = np.diag(variances)
    p = np.empty(len(evaluations))
    for i, eval_i in enumerate(evaluations):
        if noise_ceil_var is None:
            var_i = var[i]
        elif (isinstance(noise_ceil_var, np.ndarray)
              and noise_ceil_var.size > 1):
            var_i = var[i] - 2 * noise_ceil_var[i] + noise_ceil_var[-1]
        else:  # hope that noise_ceil_var is a scalar
            var_i = var[i] + noise_ceil_var
        t = (eval_i - noise_ceil) / np.sqrt(var_i)
        p[i] = 2 * (1 - stats.t.cdf(np.abs(t), dof))
    return p


def default_k_pattern(n_pattern):
    """ the default number of pattern divisions for crossvalidation
    minimum number of patterns is 3*k_pattern. Thus for n_pattern <=9 this
    returns 2. From there it grows gradually until 5 groups are made for 40
    patterns. From this point onwards the number of groups is kept at 5.

    bootstrapped crossvalidation also uses this function to set k, but scales
    n_rdm to the expected proportion of samples retained when bootstrapping
    (1-np.exp(-1))
    """
    if n_pattern < 12:
        k_pattern = 2
    elif n_pattern < 24:
        k_pattern = 3
    elif n_pattern < 40:
        k_pattern = 4
    else:
        k_pattern = 5
    return k_pattern


def default_k_rdm(n_rdm):
    """ the default number of rdm groupsfor crossvalidation
    minimum number of subjects is k_rdm. We switch to more groups whenever
    the groups all contain more rdms, e.g. we make 3 groups of 2 instead of
    2 groups of 3. We follow this scheme until we reach 5 groups of 4.
    From there on this function returns 5 groups forever.

    bootstrapped crossvalidation also uses this function to set k, but scales
    n_rdm to the expected proportion of samples retained when bootstrapping
    (1-np.exp(-1))
    """
    if n_rdm < 6:
        k_rdm = 2
    elif n_rdm < 12:
        k_rdm = 3
    elif n_rdm < 20:
        k_rdm = 4
    else:
        k_rdm = 5
    return k_rdm
