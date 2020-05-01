#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 14:04:52 2020
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pyrsa.util.inference_util import pair_tests
from pyrsa.util.rdm_utils import batch_to_vectors


def plot_model_comparison(result, alpha=0.05, plot_pair_tests=False,
                          sort=True, error_bars='SEM', eb_alpha=0.05):
    """ plots the results of a model comparison
    Input should be a results object with model evaluations
    evaluations, which uses the bootstrap samples for confidence intervals
    and significance tests and averages over all trailing dimensions
    like cross-validation folds
    Args:
        result(pyrsa.inference.result.Result): model evaluation result
    Returns:
        ---
    """
    # Preparations
    evaluations = result.evaluations
    models = result.models
    noise_ceiling = result.noise_ceiling
    method = result.method
    while len(evaluations.shape) > 2:
        evaluations = np.nanmean(evaluations, axis=-1)
    evaluations = evaluations[~np.isnan(evaluations[:, 0])]
    evaluations = 1 - evaluations
    mean = np.mean(evaluations, axis=0)
    n_models = evaluations.shape[1]
    if sort:
        idx = np.flip(np.argsort(mean))
        mean = mean[idx]
        evaluations = evaluations[:, idx]
        models = [models[i] for i in idx]
    if error_bars == 'CI':
        errorbar_low = -(np.quantile(evaluations, eb_alpha / 2, axis=0)
                         - mean)
        errorbar_high = (np.quantile(evaluations, 1 - (eb_alpha / 2), axis=0)
                         - mean)
    elif error_bars == 'SEM':
        errorbar_low = np.std(evaluations, axis=0)
        errorbar_high = np.std(evaluations, axis=0)
    noise_ceiling = 1 - np.array(noise_ceiling)
    # Plot bars
    l, b, w, h = 0.15, 0.15, 0.8, 0.8
    h_pairTests = 0.5
    if plot_pair_tests:
        plt.figure(figsize=(12.5, 10))
        ax = plt.axes((l, b, w, h*(1-h_pairTests)))
        axbar = plt.axes((l, b + h * (1 - h_pairTests), w,
                          h * h_pairTests * 0.7))
    else:
        plt.figure(figsize=(12.5, 10))
        ax = plt.axes((l, b, w, h))
    if noise_ceiling is not None:
        noise_min = np.nanmean(noise_ceiling[0])
        noise_max = np.nanmean(noise_ceiling[1])
        noiserect = patches.Rectangle((-0.5, noise_min), len(mean),
                                      noise_max - noise_min, linewidth=1,
                                      edgecolor=[0.5, 0.5, 0.5, 0.3],
                                      facecolor=[0.5, 0.5, 0.5, 0.3])
        ax.add_patch(noiserect)
    ax.bar(np.arange(evaluations.shape[1]), mean, color=[0, 0.4, 0.9, 1])
    ax.errorbar(np.arange(evaluations.shape[1]), mean,
                yerr=[errorbar_low, errorbar_high], fmt='none', ecolor='k',
                capsize=0, linewidth=3)
    # Floating axes
    ytoptick = np.ceil(min(1,ax.get_ylim()[1]) * 10) / 10
    ax.set_yticks(np.arange(0, ytoptick + 1e-6, step=0.1))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xticks(np.arange(n_models))
    ax.spines['left'].set_bounds(0, ytoptick)
    ax.spines['bottom'].set_bounds(0, n_models - 1)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    plt.rc('ytick', labelsize=18)
    # Axis labels
    fs = 20
    if models is not None:
        ax.set_xticklabels([m.name for m in models], fontsize=18,
                           rotation=45)
    if method == 'cosine':
        ax.set_ylabel('RDM prediction accuracy\n[mean cosine similarity]', fontsize=fs)
    if method == 'cosine_cov' or method == 'whitened cosine':
        ax.set_ylabel('RDM prediction accuracy\n[mean whitened-RDM cosine]', fontsize=fs)
    elif method == 'Spearman' or method == 'spearman':
        ax.set_ylabel('RDM prediction accuracy\n[mean Spearman r rank correlation]', fontsize=fs)
    elif method == 'corr' or method == 'Pearson' or method == 'pearson':
        ax.set_ylabel('RDM prediction accuracy\n[mean Pearson r correlation]', fontsize=fs)
    elif method == 'corr_cov':
        ax.set_ylabel('RDM prediction accuracy\n[mean whitened-RDM Pearson r correlation]', fontsize=fs)
    elif method == 'kendall' or method == 'tau-b':
        ax.set_ylabel('RDM prediction accuracy\n[mean Kendall tau-b rank correlation]', fontsize=fs)
    elif method == 'tau-a':
        ax.set_ylabel('RDM prediction accuracy\n[mean Kendall tau-a rank correlation]', fontsize=fs)
    # Pairwise model comparisons
    if plot_pair_tests:
        model_comp_descr = 'Model comparisons: two-tailed, '
        res = pair_tests(evaluations)
        n_tests = int((n_models**2-n_models)/2)
        if plot_pair_tests == 'Bonferroni' or plot_pair_tests == 'FWER':
            significant = res < (alpha / n_tests)
            model_comp_descr = (model_comp_descr
                                + 'p < {:3.3f}'.format(alpha)
                                + ', Bonferroni-corrected for '
                                + str(n_tests)
                                + ' model-pair comparisons')
        elif plot_pair_tests == 'FDR':
            ps = batch_to_vectors(np.array([res]))[0][0]
            ps = np.sort(ps)
            criterion = alpha * (np.arange(ps.shape[0]) + 1) / ps.shape[0]
            k_ok = ps < criterion
            if np.any(k_ok):
                k_max = np.max(np.where(ps < criterion)[0])
                crit = criterion[k_max]
            else:
                crit = 0
            significant = res < crit
            model_comp_descr = (model_comp_descr +
                                'FDR q < {:3.3f}'.format(alpha) +
                                ' (' + str(n_tests) +
                                ' model-pair comparisons)')
        else:
            significant = res < alpha
            model_comp_descr = (model_comp_descr +
                                'p < {:3.3f}'.format(alpha) +
                                ', uncorrected (' + str(n_tests) +
                                ' model-pair comparisons)')
        k = 1
        for i in range(significant.shape[0]):
            k += 1
            for j in range(i + 1, significant.shape[0]):
                if significant[i, j]:
                    axbar.plot((i, j), (k, k), 'k-', linewidth=2)
                    k += 1
        xlim = ax.get_xlim()
        axbar.set_xlim(xlim)
        axbar.set_axis_off()
        axbar.set_ylim((0, n_tests+n_models))
        if result.cv_method == 'bootstrap_rdm':
            model_comp_descr = model_comp_descr + '\nInference by bootstrap resampling of subjects.'
        elif result.cv_method == 'bootstrap_pattern':
            model_comp_descr = model_comp_descr + '\nInference by bootstrap resampling of experimental conditions.'
        elif result.cv_method == 'bootstrap':
            model_comp_descr = model_comp_descr + '\nInference by bootstrap resampling of subjects and experimental conditions.'
        model_comp_descr = model_comp_descr + '\nError bars indicate the'
        if error_bars == 'CI':
            model_comp_descr = (model_comp_descr +
                                ' {:3.0f}'.format(round(1-eb_alpha)*100) +
                                '% confidence interval.')
        elif error_bars == 'SEM':
            model_comp_descr = (model_comp_descr +
                                ' standard error of the mean.')
        axbar.set_title(model_comp_descr)