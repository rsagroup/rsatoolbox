#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 14:04:52 2020

@author: heiko
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_model_comparison(evaluations, models=None, eb_alpha=0.05):
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
    plt.bar(np.arange(evaluations.shape[1]), mean)
    plt.errorbar(np.arange(evaluations.shape[1]), mean,
                 yerr=[errorbar_low, errorbar_high], fmt='none', ecolor='k')
