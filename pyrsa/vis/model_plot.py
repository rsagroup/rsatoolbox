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


def plot_model_comparison(evaluations, models=None, eb_alpha=0.05,
                          noise_ceiling=None, plot_pair_tests=False,
                          method='cosine'):
    """ plots the results of a model comparison
    Input should be a [bootstrap samples x models x ...] array of model 
    evaluations, which uses the bootstrap samples for confidence intervals
    and significance tests and averages over all trailing dimensions 
    like cross-validation folds

    Args:
        evaluations(numpy.ndarray): model performances

    Returns:
        ---

    """
    while len(evaluations.shape)>2:
        evaluations = np.mean(evaluations, axis=-1)
    mean = np.mean(evaluations, axis=0)
    errorbar_low = -(np.quantile(evaluations, eb_alpha / 2, axis=0)
                     - mean)
    errorbar_high = (np.quantile(evaluations, 1 - (eb_alpha / 2), axis=0)
                     - mean)
    # plotting start
    if pair_tests:
        plt.figure(figsize=(7.5,10))
        ax = plt.axes((0.05,0.05, 0.9, 0.9*0.75))
        axbar = plt.axes((0.05,0.75, 0.9, 0.9*0.2))
    else:
        plt.figure(figsize=(7.5,7.5))
        ax = plt.axes((0.05,0.05,0.9,0.9))
    if noise_ceiling is not None:
        noise_min = np.mean(noise_ceiling[0])
        noise_max = np.mean(noise_ceiling[1])
        noiserect = patches.Rectangle((-0.5, noise_min), len(mean),
                                      noise_max - noise_min, linewidth=1,
                                      edgecolor=[0.25,0.25,1,0.4],
                                      facecolor=[0.25,0.25,1,0.4])
        ax.add_patch(noiserect)
    ax.bar(np.arange(evaluations.shape[1]), mean)
    ax.errorbar(np.arange(evaluations.shape[1]), mean,
                yerr=[errorbar_low, errorbar_high], fmt='none', ecolor='k')
    _,ymax = ax.get_ylim()
    ax.set_ylim(top = max(ymax,noise_max))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xticks(np.arange(len(mean)))
    if models is not None:
        ax.set_xticklabels([m.name for m in models])
    if method == 'cosine':
        ax.set_ylabel('cosine distance', fontsize=18)
    elif method == 'spearman':
        ax.set_ylabel('1 - Spearman rank correlation',fontsize=18)
    elif method == 'corr':
        ax.set_ylabel('1 - Pearson correlation',fontsize=18)
        
    if plot_pair_tests:
        res = pair_tests(evaluations)
        significant = res < eb_alpha
        k = 0
        for i in range(significant.shape[0]):
            for j in range(significant.shape[0]):
                if significant[i,j]:
                    axbar.plot((i,j),(k,k),'k-',linewidth=2)
                    k = k+1
        xlim = ax.get_xlim()
        axbar.set_xlim(xlim)
        axbar.set_axis_off()
        axbar.set_ylim((-0.5,k))
