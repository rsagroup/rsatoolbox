#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 14:04:52 2020

@author: heiko
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
    evaluations = result.evaluations
    models = result.models
    noise_ceiling = result.noise_ceiling
    method = result.method
    while len(evaluations.shape)>2:
        evaluations = np.nanmean(evaluations, axis=-1)
    evaluations = evaluations[~np.isnan(evaluations[:,0])]
    evaluations = 1 - evaluations
    mean = np.mean(evaluations, axis=0)
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
    noise_ceiling = 1 - noise_ceiling
    # plotting start
    if plot_pair_tests:
        plt.figure(figsize=(12.5, 10))
        ax = plt.axes((0.05, 0.05, 0.9, 0.9 * 0.75))
        axbar = plt.axes((0.05, 0.75, 0.9, 0.9 * 0.2))
    else:
        plt.figure(figsize=(12.5, 7.5))
        ax = plt.axes((0.05, 0.05, 0.9, 0.9))
    if noise_ceiling is not None:
        noise_min = np.nanmean(noise_ceiling[0])
        noise_max = np.nanmean(noise_ceiling[1])
        noiserect = patches.Rectangle((-0.5, noise_min), len(mean),
                                      noise_max - noise_min, linewidth=1,
                                      edgecolor=[0.25, 0.25, 1, 0.4],
                                      facecolor=[0.25, 0.25, 1, 0.4])
        ax.add_patch(noiserect)
    ax.bar(np.arange(evaluations.shape[1]), mean)
    ax.errorbar(np.arange(evaluations.shape[1]), mean,
                yerr=[errorbar_low, errorbar_high], fmt='none', ecolor='k',
                capsize=0, linewidth=4)
    _,ymax = ax.get_ylim()
    ax.set_ylim(top = max(ymax, noise_max))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xticks(np.arange(len(mean)))
    plt.rc('ytick', labelsize=18)
    if models is not None:
        ax.set_xticklabels([m.name for m in models], fontsize=18,
                           rotation=75)
    if method == 'cosine':
        ax.set_ylabel('cosine distance', fontsize=24)
    if method == 'cosine_cov':
        ax.set_ylabel('Cov-weighted cosine distance', fontsize=24)
    elif method == 'spearman':
        ax.set_ylabel('Spearman rank correlation', fontsize=24)
    elif method == 'corr':
        ax.set_ylabel('Pearson correlation', fontsize=24)
    elif method == 'corr_cov':
        ax.set_ylabel('Cov-weighted correlation', fontsize=24)
    elif method == 'kendall' or method == 'tau-b':
        ax.set_ylabel('Kendall-Tau', fontsize=24)
    elif method == 'tau-a':
        ax.set_ylabel('Kendall-Tau A', fontsize=24)
    if plot_pair_tests:
        res = pair_tests(evaluations)
        if plot_pair_tests == 'Bonferroni' or plot_pair_tests == 'FWER':
            significant = res < (alpha / evaluations.shape[1])
        elif plot_pair_tests == 'FDR':
            ps = batch_to_vectors(np.array([res]))[0][0]
            ps = np.sort(ps)
            criterion = alpha * (np.arange(ps.shape[0]) + 1) / ps.shape[0]
            k_ok = ps < criterion
            if np.any(k_ok):
                k_max = np.max(np.where(ps<criterion)[0])
                crit = criterion[k_max]
            else:
                crit = 0
            significant = res < crit
        else:
            significant = res < alpha
        k = 0
        for i in range(significant.shape[0]):
            for j in range(i+1,significant.shape[0]):
                if significant[i,j]:
                    axbar.plot((i,j), (k,k), 'k-', linewidth=2)
                    k = k+1
        xlim = ax.get_xlim()
        axbar.set_xlim(xlim)
        axbar.set_axis_off()
        axbar.set_ylim((-0.5,k))
