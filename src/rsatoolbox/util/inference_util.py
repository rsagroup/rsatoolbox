#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inference module utilities
"""
from __future__ import annotations
from collections.abc import Iterable
from typing import TYPE_CHECKING, Optional
import warnings
import numpy as np
from scipy import stats
from scipy.stats import rankdata, wilcoxon
from scipy.stats import t as tdist
from rsatoolbox.model import Model
from rsatoolbox.rdm import RDMs
from .matrix import pairwise_contrast
from .rdm_utils import batch_to_matrices
if TYPE_CHECKING:
    from numpy.typing import NDArray


def input_check_model(models, theta=None, fitter=None, N=1):
    """ Checks whether model related inputs to evaluations are valid and
    generates an evaluation-matrix of fitting size.

    Args:
        model : [list of] rsatoolbox.rdm.RDMs
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
        raise ValueError('model should be an rsatoolbox.model.Model or a list of'
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


def pool_rdm(rdms, method: str = 'cosine'):
    """pools multiple RDMs into the one with maximal performance under a given
    evaluation metric
    rdm_descriptors of the generated rdms are empty

    Args:
        rdms (rsatoolbox.rdm.RDMs):
            RDMs to be pooled
        method : String, optional
            Which comparison method to optimize for. The default is 'cosine'.

    Returns:
        rsatoolbox.rdm.RDMs: the pooled RDM, i.e. a RDM with maximal performance
            under the chosen method

    """
    rdm_vec = rdms.get_vectors()
    if method == 'euclid':
        rdm_vec = _nan_mean(rdm_vec)
    elif method == 'neg_riem_dist':
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
    elif method in ('spearman', 'rho-a'):
        rdm_vec = np.array([_nan_rank_data(v) for v in rdm_vec])
        rdm_vec = _nan_mean(rdm_vec)
    elif method == 'rho-a':
        rdm_vec = np.array([_nan_rank_data(v) for v in rdm_vec])
        rdm_vec = _nan_mean(rdm_vec)
    elif method in ('kendall', 'tau-b'):
        warnings.warn('Noise ceiling for tau based on averaged ranks!')
        rdm_vec = np.array([_nan_rank_data(v) for v in rdm_vec])
        rdm_vec = _nan_mean(rdm_vec)
    elif method == 'tau-a':
        warnings.warn('Noise ceiling for tau based on averaged ranks!')
        rdm_vec = np.array([_nan_rank_data(v) for v in rdm_vec])
        rdm_vec = _nan_mean(rdm_vec)
    else:
        raise ValueError('Unknown RDM comparison method requested!')
    return RDMs(rdm_vec,
                dissimilarity_measure=rdms.dissimilarity_measure,
                descriptors=rdms.descriptors,
                rdm_descriptors=None,
                pattern_descriptors=rdms.pattern_descriptors)


def _nan_mean(rdm_vector: NDArray) -> NDArray:
    """ takes the average over a rdm_vector with nans for masked entries
    without a warning

    Args:
        rdm_vector(numpy.ndarray): set of rdm_vectors to be averaged

    Returns:
        rdm_mean(numpy.ndarray): the mean rdm

    """
    nan_idx = ~np.isnan(rdm_vector[0])
    mean_values = np.mean(rdm_vector[:, nan_idx], axis=0)
    rdm_mean = np.empty((1, rdm_vector.shape[1]))
    rdm_mean.fill(np.nan)
    rdm_mean[:, nan_idx] = mean_values
    return rdm_mean


def _nan_rank_data(rdm_vector: NDArray) -> NDArray:
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


def all_tests(
        evaluations: NDArray,
        noise_ceil: NDArray,
        test_type: str = 't-test',
        model_var: Optional[NDArray] = None,
        diff_var: Optional[NDArray] = None,
        noise_ceil_var: Optional[NDArray] = None,
        dof: int = 1):
    """wrapper running all tests necessary for the model plot
    -> pairwise tests, tests against 0 and against noise ceiling


    Args:
        evaluations (numpy.ndarray):
            model evaluations to be compared
            (should be 3D: bootstrap x models x subjects or repeats)
        noise_ceil (numpy.ndarray):
            noise_ceiling estimate(s) to compare against
        test_type(String):
            't-test' : t-test bases tests using variances
            'bootstrap' : Direct bootstrap sample based tests
            'ranksum' : Wilcoxon signed rank-sum tests
        model_var, diff_var, noise_ceil_var:
            variance estimates from the results object.
            These correspond to the variances of the individual model evals,
            pairs of model evals and differences from the noise ceiling.

    Returns:
        numpy.ndarrays: p_pairwise, p_zero, p_noise

    """
    if test_type == 't-test':
        p_pairwise = t_tests(evaluations, diff_var, dof=dof)
        p_zero = t_test_0(evaluations, model_var, dof=dof)
        p_noise = t_test_nc(evaluations, noise_ceil_var[:, 0],
                            np.nanmean(noise_ceil[0]), dof)
    elif test_type == 'bootstrap':
        if len(noise_ceil.shape) > 1:
            noise_lower_bs = noise_ceil[0]
            noise_lower_bs.shape = (noise_ceil.shape[0], 1)
        else:
            noise_lower_bs = noise_ceil[0].reshape(1, 1)
        p_pairwise = bootstrap_pair_tests(evaluations)
        p_zero = ((evaluations <= 0).sum(axis=0) + 1) / evaluations.shape[0]
        diffs = noise_lower_bs - evaluations
        p_noise = ((diffs <= 0).sum(axis=0) + 1) / evaluations.shape[0]
    elif test_type == 'ranksum':
        noise_c = np.nanmean(noise_ceil[0])
        p_pairwise = ranksum_pair_test(evaluations)
        p_zero = ranksum_value_test(evaluations, 0)
        p_noise = ranksum_value_test(evaluations, noise_c)
    else:
        raise ValueError('test_type not recognized.\n'
                         + 'Options are: t-test, bootstrap, ranksum')
    return p_pairwise, p_zero, p_noise


def pair_tests(
        evaluations: NDArray,
        test_type: str = 't-test',
        diff_var: Optional[NDArray] = None,
        dof: int = 1):
    """wrapper running pair tests

    Args:
        evaluations (numpy.ndarray):
            model evaluations to be compared
            (should be 3D: bootstrap x models x subjects or repeats)
        test_type(Strinng):
            't-test' : t-test bases tests using variances
            'bootstrap' : Direct bootstrap sample based tests
            'ranksum' : Wilcoxon signed rank-sum tests

    Returns:
        numpy.ndarray: p_pairwise

    """
    if test_type == 't-test':
        p_pairwise = t_tests(evaluations, diff_var, dof=dof)
    elif test_type == 'bootstrap':
        p_pairwise = bootstrap_pair_tests(evaluations)
    elif test_type == 'ranksum':
        p_pairwise = ranksum_pair_test(evaluations)
    else:
        raise ValueError('test_type not recognized.\n'
                         + 'Options are: t-test, bootstrap, ranksum')
    return p_pairwise


def zero_tests(evaluations, test_type='t-test',
               model_var=None, dof=1):
    """wrapper running tests against 0

    Args:
        evaluations (numpy.ndarray):
            model evaluations to be compared
            (should be 3D: bootstrap x models x subjects or repeats)
        test_type(Strinng):
            't-test' : t-test bases tests using variances
            'bootstrap' : Direct bootstrap sample based tests
            'ranksum' : Wilcoxon signed rank-sum tests

    Returns:
        numpy.ndarray: p_zero

    """
    if test_type == 't-test':
        p_zero = t_test_0(evaluations, model_var, dof=dof)
    elif test_type == 'bootstrap':
        p_zero = ((evaluations <= 0).sum(axis=0) + 1) / evaluations.shape[0]
    elif test_type == 'ranksum':
        p_zero = ranksum_value_test(evaluations, 0)
    else:
        raise ValueError('test_type not recognized.\n'
                         + 'Options are: t-test, bootstrap, ranksum')
    return p_zero


def nc_tests(evaluations, noise_ceil, test_type='t-test',
             noise_ceil_var=None, dof=1):
    """wrapper running tests against noise ceiling

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
        noise_ceil_var:
            variance estimate from the results object.
            These correspond to the variances of the
            differences from the noise ceiling.

    Returns:
        numpy.ndarrays: p_noise

    """
    if test_type == 't-test':
        p_noise = t_test_nc(evaluations, noise_ceil_var[:, 0],
                            np.nanmean(noise_ceil[0]), dof)
    elif test_type == 'bootstrap':
        if len(noise_ceil.shape) > 1:
            noise_lower_bs = noise_ceil[0]
            noise_lower_bs.shape = (noise_ceil.shape[0], 1)
        else:
            noise_lower_bs = noise_ceil[0].reshape(1, 1)
        diffs = noise_lower_bs - evaluations
        p_noise = ((diffs <= 0).sum(axis=0) + 1) / evaluations.shape[0]
    elif test_type == 'ranksum':
        noise_c = np.nanmean(noise_ceil[0])
        p_noise = ranksum_value_test(evaluations, noise_c)
    else:
        raise ValueError('test_type not recognized.\n'
                         + 'Options are: t-test, bootstrap, ranksum')
    return p_noise


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


def bootstrap_pair_tests(evaluations):
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
        evaluations = np.nanmean(evaluations, axis=-1)
    for i_model in range(evaluations.shape[1] - 1):
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

    Args:
        evaluations (numpy.ndarray):
            model evaluations to be tested, typically from a results object
        variances (numpy.ndarray):
            vector of the variances of model evaluation differences
        dof (integer):
            degrees of freedom used for the test (default=1)

    Returns:
        numpy.ndarray: matrix of p-values for the test

    """
    if variances is None:
        raise ValueError('No variance estimates provided for t_test!')
    n_model = evaluations.shape[1]
    evaluations = np.nanmean(evaluations, 0)
    while evaluations.ndim > 1:
        evaluations = np.nanmean(evaluations, axis=-1)
    C = pairwise_contrast(np.arange(n_model))
    diffs = C @ evaluations
    t = diffs / np.sqrt(np.maximum(variances, np.finfo(float).eps))
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
        dof (integer):
            degrees of freedom used for the test (default=1)

    Returns:
        numpy.ndarray: p-values for the raw t-test of each model against 0.

    """
    if variances is None:
        raise ValueError('No variance estimates provided for t_test!')
    evaluations = np.nanmean(evaluations, 0)
    while evaluations.ndim > 1:
        evaluations = np.nanmean(evaluations, axis=-1)
    t = evaluations / np.sqrt(np.maximum(variances, np.finfo(float).eps))
    p = 1 - stats.t.cdf(t, dof)
    return p


def t_test_nc(evaluations, variances, noise_ceil, dof=1):
    """
    t-tests against noise_ceiling.
    Technically this can be used to test evaluations against any fixed
    number.

    Args:
        evaluations (numpy.ndarray):
            model evaluations to be tested, typically from a results object
        variances (numpy.ndarray):
            variance estimates for the comparisons to the noise ceiling
        noise_ceil (float):
            the average noise ceiling to test against.
        noise_ceil_var (numpy.ndarray):
            variance or covariance of the noise ceiling
        dof (integer):
            degrees of freedom used for the test (default=1)

    Returns:
        numpy.ndarray: p-values for the raw t-test of each model against
        the noise ceiling.

    """
    if variances is None:
        raise ValueError('No variance estimates provided for t_test!')
    evaluations = np.nanmean(evaluations, 0)
    while evaluations.ndim > 1:
        evaluations = np.nanmean(evaluations, axis=-1)
    p = np.empty(len(evaluations))
    for i, eval_i in enumerate(evaluations):
        t = (eval_i - noise_ceil) / np.sqrt(
            np.maximum(variances[i], np.finfo(float).eps))
        p[i] = 2 * (1 - stats.t.cdf(np.abs(t), dof))
    return p


def extract_variances(
        variance,
        nc_included: bool = True,
        n_rdm: Optional[int] = None,
        n_pattern: Optional[int] = None):
    """ extracts the variances for the individual model evaluations,
    differences between model evaluations and for the comparison to
    the noise ceiling

    for 1D arrays we assume a diagonal covariance is meant

    for 2D arrays this is taken as the covariance of the model evals

    for 3D arrays we assume this is the result of a dual bootstrap and
    perform the correction. Then there should be three covariances given
    from double, rdm & pattern bootstrap in that order.

    nc_included=True jields the result if the last two columns correspond
    to the noise ceiling results

    nc_included=False assumes that the noise ceiling is fixed instead.

    To get the more accurate estimates that take into account
    the number of subjects and/or the numbers of stimuli
    can be passed as n_rdm and n_pattern respectively.
    This function corrects for all ns that are passed. If you bootstrapped
    only one factor only pass the N for that factor!
    """
    if variance.ndim == 0:
        variance = np.array([variance])
    if variance.ndim == 1:
        # model evaluations assumed independent
        if nc_included:
            C = pairwise_contrast(np.arange(variance.shape[0] - 2))
            model_variances = variance[:-2]
            nc_variances = np.expand_dims(model_variances, -1) \
                + np.expand_dims(variance[-2:], 0)
            diff_variances = np.diag(C @ np.diag(variance[:-2]) @ C.T)
        else:
            C = pairwise_contrast(np.arange(variance.shape[0]))
            model_variances = variance
            nc_variances = np.array([variance, variance]).T
            diff_variances = np.diag(C @ np.diag(variance) @ C.T)
        model_variances = _correct_1d(model_variances, n_pattern, n_rdm)
        nc_variances = _correct_1d(nc_variances, n_pattern, n_rdm)
        diff_variances = _correct_1d(diff_variances, n_pattern, n_rdm)
    elif variance.ndim == 2:
        # a single covariance matrix
        if nc_included:
            C = pairwise_contrast(np.arange(variance.shape[0] - 2))
            model_variances = np.diag(variance)[:-2]
            nc_variances = np.expand_dims(model_variances, -1) \
                - 2 * variance[:-2, -2:] \
                + np.expand_dims(np.diag(variance[-2:, -2:]), 0)
            diff_variances = np.diag(C @ variance[:-2, :-2] @ C.T)
        else:
            C = pairwise_contrast(np.arange(variance.shape[0]))
            model_variances = np.diag(variance)
            nc_variances = np.array([model_variances, model_variances]).T
            diff_variances = np.diag(C @ variance @ C.T)
        model_variances = _correct_1d(model_variances, n_pattern, n_rdm)
        nc_variances = _correct_1d(nc_variances, n_pattern, n_rdm)
        diff_variances = _correct_1d(diff_variances, n_pattern, n_rdm)
    elif variance.ndim == 3:
        # general transform for multiple covariance matrices
        if nc_included:
            C = pairwise_contrast(np.arange(variance.shape[1] - 2))
            model_variances = np.einsum('ijj->ij', variance)[:, :-2]
            nc_variances = np.expand_dims(model_variances, -1) \
                - 2 * variance[:, :-2, -2:] \
                + np.expand_dims(np.einsum('ijj->ij',
                                           variance[:, -2:, -2:]), 1)
            # np.diag(C@variances@C.T)
            diff_variances = np.einsum(
                'ij,kjl,il->ki', C, variance[:, :-2, :-2], C)
        else:
            C = pairwise_contrast(np.arange(variance.shape[1]))
            model_variances = np.einsum('ijj->ij', variance)
            nc_variances = np.array([model_variances, model_variances]
                                    ).transpose(1, 2, 0)
            diff_variances = np.einsum('ij,kjl,il->ki', C, variance, C)
        # dual bootstrap variance estimate from 3 covariance matrices
        model_variances = _dual_bootstrap(model_variances, n_rdm, n_pattern)
        nc_variances = _dual_bootstrap(nc_variances, n_rdm, n_pattern)
        diff_variances = _dual_bootstrap(diff_variances, n_rdm, n_pattern)
    return model_variances, diff_variances, nc_variances


def _correct_1d(
        variance: NDArray,
        n_pattern: Optional[int] = None,
        n_rdm: Optional[int] = None):
    if (n_pattern is not None) and (n_rdm is not None):
        # uncorrected dual bootstrap?
        n = min(n_rdm, n_pattern)
    elif n_pattern is not None:
        n = n_pattern
    elif n_rdm is not None:
        n = n_rdm
    else:
        n = None
    if n is not None:
        variance = (n / (n - 1)) * variance
    return variance


def get_errorbars(model_var, evaluations, dof, error_bars='sem',
                  test_type='t-test'):
    """ computes errorbars for the model-evaluations from a results object

    Args:
        results : rsatoolbox.inference.Results
            the results object
        eb_type : str, optional
            which errorbars to compute

    Returns:
        limits (np.array)
    """
    if model_var is None:
        return np.full((2, evaluations.shape[1]), np.nan)
    if error_bars.lower() == 'sem':
        errorbar_low = np.sqrt(np.maximum(model_var, 0))
        errorbar_high = np.sqrt(np.maximum(model_var, 0))
    elif error_bars[0:2].lower() == 'ci':
        if len(error_bars) == 2:
            CI_percent = 95.0
        else:
            CI_percent = float(error_bars[2:])
        prop_cut = (1 - CI_percent / 100) / 2
        if test_type == 'bootstrap':
            n_models = evaluations.shape[1]
            framed_evals = np.concatenate(
                (np.tile(np.array((-np.inf, np.inf)).reshape(2, 1),
                         (1, n_models)),
                 evaluations),
                axis=0)
            perf = np.nanmean(evaluations, 0)
            while perf.ndim > 1:
                perf = np.nanmean(perf, -1)
            errorbar_low = -(np.quantile(framed_evals, prop_cut, axis=0)
                             - perf)
            errorbar_high = (np.quantile(framed_evals, 1 - prop_cut,
                                         axis=0)
                             - perf)
        else:
            std_eval = np.sqrt(np.maximum(model_var, 0))
            errorbar_low = std_eval \
                * tdist.ppf(prop_cut, dof)
            errorbar_high = std_eval \
                * tdist.ppf(prop_cut, dof)
    else:
        raise ValueError('computing errorbars: Argument ' +
                         'error_bars is incorrectly defined as '
                         + str(error_bars) + '.')
    limits = np.stack((errorbar_low, errorbar_high))
    if np.isnan(limits).any() or (abs(limits) == np.inf).any():
        raise ValueError(
            'computing errorbars: Too few bootstrap samples for the ' +
            'requested confidence interval: ' + error_bars + '.')
    return limits


def _dual_bootstrap(variances, n_rdm=None, n_pattern=None):
    """ helper function to perform the dual bootstrap

    Takes a 3x... array of variances and computes the corrections assuming:
    variances[0] are the variances in the double bootstrap
    variances[1] are the variances in the rdm bootstrap
    variances[2] are the variances in the pattern bootstrap

    If both n_rdm and n_pattern are given this uses
    the more accurate small sample formula.
    """
    if n_rdm is None or n_pattern is None:
        variance = 2 * (variances[1] + variances[2]) \
            - variances[0]
        variance = np.maximum(np.maximum(
            variance, variances[1]), variances[2])
        variance = np.minimum(
            variance, variances[0])
    else:
        variance = (
            (n_rdm / (n_rdm - 1)) * variances[1]
            + (n_pattern / (n_pattern - 1)) * variances[2]
            - ((n_pattern * n_rdm / (n_pattern - 1) / (n_rdm - 1))
               * (variances[0] - variances[1] - variances[2])))
        variance = np.maximum(np.maximum(
            variance,
            (n_rdm / (n_rdm - 1)) * variances[1]),
            (n_pattern / (n_pattern - 1)) * variances[2])
        variance = np.minimum(
            variance, variances[0])
    return variance


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
