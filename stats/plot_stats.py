#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 22:29:52 2020

@author: heiko
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from matplotlib import rcParams
import seaborn as sns
import pandas as pd
import scipy.stats
from helpers import get_fname_base
import rsatoolbox
import pathlib
import nn_simulations as dnn
from helpers import get_stimuli_ecoset
from stats import estimate_betas, get_residuals

rcParams.update({'figure.autolayout': True})


def plot_saved_dnn_average(layer=2, sd=3,
                           n_voxel=100, n_subj=10, simulation_folder='test',
                           n_sim=100, n_repeat=2, duration=5, pause=1,
                           endzeros=25, use_cor_noise=True, resolution=2,
                           sigma_noise=2, ar_coeff=.5, modelType='fixed',
                           model_rdm='averagetrue', n_stimuli=92,
                           rdm_comparison='cosine', n_layer=12, n_fold=5,
                           rdm_type='crossnobis', fname_base=None):
    if fname_base is None:
        fname_base = get_fname_base(simulation_folder=simulation_folder,
                                    layer=layer, n_voxel=n_voxel,
                                    n_subj=n_subj, duration=duration,
                                    n_repeat=n_repeat, sd=sd,
                                    pause=pause, endzeros=endzeros,
                                    use_cor_noise=use_cor_noise,
                                    resolution=resolution,
                                    sigma_noise=sigma_noise,
                                    ar_coeff=ar_coeff)
    assert os.path.isdir(fname_base), 'simulated data not found!'
    scores = np.load(fname_base + 'scores_%s_%s_%s_%s_%d_%d.npy' % (
        rdm_type, modelType, model_rdm, rdm_comparison, n_stimuli, n_fold))
    noise_ceilings = np.load(fname_base + 'noisec_%s_%s_%s_%s_%d_%d.npy' % (
        rdm_type, modelType, model_rdm, rdm_comparison, n_stimuli, n_fold))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.tick_params(labelsize=12)
    for i_sim in range(n_sim):
        ax.fill_between(np.array([0.5, n_layer + 0.5]),
                        noise_ceilings[i_sim, 0],
                        noise_ceilings[i_sim, 1], alpha=1 / n_sim,
                        facecolor='blue')
    ax.plot(np.arange(n_layer) + 1 - n_fold / 20,
            np.mean(scores[:, :n_layer, :], axis=2).T, 'k.')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel('Layer', fontsize=18)
    ax.set_title('Layer %d' % layer, fontsize=28)
    if rdm_comparison == 'cosine':
        plt.ylim([0, 1])
        ax.set_ylabel('Cosine Distance', fontsize=18)
    elif rdm_comparison == 'eudlid':
        ax.set_ylabel('Euclidean Distance', fontsize=18)
    elif rdm_comparison == 'kendall-tau':
        ax.set_ylabel('Kendall Tau', fontsize=18)
    elif rdm_comparison == 'pearson':
        ax.set_ylabel('Pearson Correlation', fontsize=18)
    elif rdm_comparison == 'spearman':
        ax.set_ylabel('Spearman Rho', fontsize=18)


def plot_saved_dnn(layer=2, sd=0.05, n_voxel=100, idx=0,
                   n_subj=10, simulation_folder='sim', n_repeat=2,
                   duration=1, pause=1, endzeros=25, use_cor_noise=True,
                   resolution=2, sigma_noise=2, ar_coeff=0.5,
                   model_type='fixed_averagetrue',
                   rdm_comparison='cosine', n_layer=12, k_pattern=3, k_rdm=3,
                   rdm_type='crossnobis', n_stimuli=92, fname_base=None,
                   noise_type='eye'):
    if fname_base is None:
        fname_base = get_fname_base(simulation_folder=simulation_folder,
                                    layer=layer, n_voxel=n_voxel,
                                    n_subj=n_subj, sd=sd,
                                    n_repeat=n_repeat, duration=duration,
                                    pause=pause, endzeros=endzeros,
                                    use_cor_noise=use_cor_noise,
                                    resolution=resolution,
                                    sigma_noise=sigma_noise,
                                    ar_coeff=ar_coeff)
    assert os.path.isdir(fname_base), 'simulated data not found!'
    res_path = fname_base + 'results_%s_%s_%s_%s_%d_%d_%d' % (
        rdm_type, model_type, rdm_comparison, noise_type, n_stimuli,
        k_pattern, k_rdm)
    results = rsatoolbox.inference.load_results(res_path + '/res%04d.hdf5' % idx)
    rsatoolbox.vis.plot_model_comparison(results)


def plot_compare_to_zero(n_voxel=100, n_subj=10, n_cond=5,
                         method='corr', bootstrap='pattern',
                         folder='comp_zero', n_bin=100):
    fname = 'p_'
    if method:
        fname = fname + ('%s_' % method)
    else:
        fname = fname + '*_'
    if bootstrap:
        fname = fname + ('%s_' % bootstrap)
    else:
        fname = fname + '*_'
    if n_cond:
        fname = fname + ('%d_' % n_cond)
    else:
        fname = fname + '*_'
    if n_subj:
        fname = fname + ('%d_' % n_subj)
    else:
        fname = fname + '*_'
    if n_voxel:
        fname = fname + ('%d_' % n_voxel)
    else:
        fname = fname + '*_'
    fname = fname + '*.npy'
    n_significant = []
    n_binned = []
    bins = np.linspace(1 / n_bin, 1, n_bin)
    for p in pathlib.Path(folder).glob(fname):
        ps = np.load(p)
        n = np.empty(n_bin)
        for i_bin in range(n_bin):
            n[i_bin] = np.sum(ps <= bins[i_bin])
        n_binned.append(n)
        n_significant.append(np.sum(ps < 0.05))
    n_binned = np.array(n_binned)
    n_significant = np.array(n_significant)
    n_binned = n_binned / n_binned[:, -1].reshape(-1, 1)
    ax = plt.subplot(1, 1, 1)
    plt.plot(bins, n_binned.T)
    plt.plot([0, 1], [0, 1], 'k--')
    ax.set_aspect('equal', 'box')
    plt.xlabel('alpha')
    plt.ylabel('proportion p<alpha')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.figure()
    plt.plot(n_significant, 'k.')
    plt.plot([0 - 0.5, len(n_significant) - 0.5], [50, 50], 'k--')
    plt.ylim(bottom=0)


def plot_comp(data, alpha=0.05, save_file=None):
    """ plots comp check data
    """
    # methods = np.unique(data[:, 1])
    # boots = np.unique(data[:, 2])
    test_type_id = 10 * data[:, 2] + data[:, 3]
    test_ids = np.unique(test_type_id)
    n_subj = np.unique(data[:, 4])
    n_cond = np.unique(data[:, 5])
    n_voxel = np.unique(data[:, 6])
    # boot_noise = np.unique(data[:, 6])
    # sigmas = np.unique(data[:, 7])
    # idx = np.unique(data[:, 9])
    props = np.nan * np.empty((len(test_ids), len(n_subj), len(n_cond),
                               len(n_voxel)))
    for i_test, test in enumerate(test_ids):
        for i_subj, n_sub in enumerate(n_subj):
            for i_cond, cond in enumerate(n_cond):
                for i_vox, vox in enumerate(n_voxel):
                    dat = data[test_type_id == test, :]
                    dat = dat[dat[:, 4] == n_sub, :]
                    dat = dat[dat[:, 5] == cond, :]
                    dat = dat[dat[:, 6] == vox, :]
                    if len(dat) > 0:
                        prop = (np.sum(dat[:, 0] < alpha)
                                / len(dat))
                        props[i_test, i_subj, i_cond, i_vox] = prop
                    else:
                        props[i_test, i_subj, i_cond, i_vox] = np.nan
    # First plot: barplot + scatter for each type of bootstrap
    plt.figure()
    ax = plt.subplot(1, 1, 1)
    for i in range(len(test_ids)):
        plt.bar(i, np.mean(props[i]))
        plt.plot(np.repeat(i, props[i].size)
                 + 0.1 * np.random.randn(props[i].size),
                 props[i].flatten(), 'k.')
    plt.plot([-0.5, len(test_ids) - 0.5], [alpha, alpha], 'k--')
    plt.xticks([0, 1, 2, 3],
               ['Bootstrap\nboth', 'Bootstrap\nrdm', 'Bootstrap\npattern',
                'Wilcoxon'])
    plt.ylabel('Proportion significant', fontsize=18)
    plt.xlabel('Test type', fontsize=18)
    if save_file:
        fname = save_file + '_bars.pdf'
        plt.savefig(fname, bbox_inches='tight')
    # Second plot: plot against n_subj
    plt.figure(figsize=(3 * len(test_ids), 5))
    titles = {
        0: 'Bootstrap\nboth',
        1: 'Bootstrap\nboth, T',
        10: 'Bootstrap\nrdm',
        11: 'Bootstrap\nrdm, T',
        20: 'Bootstrap\npattern',
        21: 'Bootstrap\npattern, T',
        41: 'Bootstrap\nFormula, T',
        51: 'T-Test\n',
        52: 'Wilcoxon\n'}
    for i, t_id in enumerate(test_ids):
        ax = plt.subplot(1, len(test_ids), i + 1)
        h0 = plt.plot(np.arange(len(n_subj)) - 0.225,
                      props[i, :, 0, 0], '.',
                      color=sns.palettes.color_palette('Greens_d')[0],
                      markersize=15)
        h1 = plt.plot(np.arange(len(n_subj)) - 0.075,
                      props[i, :, 1, 0], '.',
                      color=sns.palettes.color_palette('Greens_d')[1],
                      markersize=15)
        h2 = plt.plot(np.arange(len(n_subj)) + 0.075,
                      props[i, :, 2, 0], '.',
                      color=sns.palettes.color_palette('Greens_d')[2],
                      markersize=15)
        h3 = plt.plot(np.arange(len(n_subj)) + 0.225,
                      props[i, :, 3, 0], '.',
                      color=sns.palettes.color_palette('Greens_d')[3],
                      markersize=15)
        plt.yticks([0, alpha, 2*alpha, 3*alpha], fontsize=18)
        if i == 0:
            plt.ylabel('Proportion significant', fontsize=24)
        else:
            plt.tick_params(labelleft=False)
        plt.title(titles[t_id], fontsize=18)
        if i == (len(test_ids) - 1):
            legend = plt.legend(
                [h0[0], h1[0], h2[0], h3[0]], n_cond.astype('int'),
                frameon=False, title='# of conditions', fontsize=18,
                bbox_to_anchor=(1.0, 1.0), loc=2)
            legend.get_title().set_fontsize('18')
        plt.xticks(np.arange(len(n_subj)), n_subj.astype('int'), fontsize=18)
        plt.yticks([0, alpha, 2*alpha, 3*alpha])
        plt.ylim([0, 0.15])
        plt.xlim([-1, len(n_subj)])
        plt.plot([-1, len(n_subj)], [alpha, alpha], 'k--')
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        plt.xlabel('# of rdms', fontsize=18)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    if save_file:
        fname = save_file + '_rdm.pdf'
        plt.savefig(fname, bbox_inches='tight')
    # Third plot: plot against n_pattern
    plt.figure(figsize=(3 * len(test_ids), 5))
    for i, t_id in enumerate(test_ids):
        ax = plt.subplot(1, len(test_ids), i+1)
        h0 = plt.plot(np.arange(len(n_cond)) - 0.225,
                      props[i, 0, :, 0], '.',
                      color=sns.palettes.color_palette('Blues_d')[0],
                      markersize=15)
        h1 = plt.plot(np.arange(len(n_cond)) - 0.075,
                      props[i, 1, :, 0], '.',
                      color=sns.palettes.color_palette('Blues_d')[1],
                      markersize=15)
        h2 = plt.plot(np.arange(len(n_cond)) + 0.075,
                      props[i, 2, :, 0], '.',
                      color=sns.palettes.color_palette('Blues_d')[2],
                      markersize=15)
        h3 = plt.plot(np.arange(len(n_cond)) + 0.225,
                      props[i, 3, :, 0], '.',
                      color=sns.palettes.color_palette('Blues_d')[3],
                      markersize=15)
        plt.yticks([0, alpha, 2*alpha, 3*alpha], fontsize=18)
        if i == 0:
            plt.ylabel('Proportion significant', fontsize=24)
        else:
            plt.tick_params(labelleft=False)
        plt.title(titles[t_id], fontsize=18)
        if i == (len(test_ids) - 1):
            legend = plt.legend(
                [h0[0], h1[0], h2[0], h3[0]], n_subj.astype('int'),
                frameon=False, title='# of rdms', fontsize=18,
                bbox_to_anchor=(1.0, 1.0), loc=2)
            legend.get_title().set_fontsize('18')
        plt.xticks(np.arange(len(n_cond)), n_cond.astype('int'), fontsize=18)
        plt.yticks([0, alpha, 2*alpha, 3*alpha])
        plt.ylim([0, 0.15])
        plt.xlim([-1, len(n_cond)])
        plt.plot([-1, len(n_cond)], [alpha, alpha], 'k--')
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        plt.xlabel('# of conditions', fontsize=18)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    if save_file:
        fname = save_file + '_pattern.pdf'
        plt.savefig(fname, bbox_inches='tight')


def plot_eco(simulation_folder='sim_eco', variation='both', savefig=False):
    labels = pd.read_csv(os.path.join(simulation_folder, 'labels.csv'))
    means = np.load(os.path.join(simulation_folder, 'means.npy'))
    stds = np.load(os.path.join(simulation_folder, 'stds.npy'))
    means = means[:len(labels)]
    # remove nan entries
    idx_nan = ~np.any(np.isnan(means[:, :, 0]), axis=1)
    labels = labels[list(idx_nan)]
    means = means[idx_nan]
    stds = stds[idx_nan]
    # get correct variation types
    variations = labels['variation']
    if variation is None:
        labels = labels[pd.isna(variations)]
        means = means[np.array(pd.isna(variations))]
        stds = stds[np.array(pd.isna(variations))]
    elif variation[:4] == 'None':
        labels = labels[pd.isna(variations)]
        means = means[np.array(pd.isna(variations))]
        stds = stds[np.array(pd.isna(variations))]
        boot_type = labels['boot_type']
        labels = labels[list(boot_type == variation[5:])]
        means = means[boot_type == variation[5:]]
        stds = stds[boot_type == variation[5:]]
    else:
        labels = labels[list(variations == variation)]
        means = means[variations == variation]
        stds = stds[variations == variation]
    true_std = np.nanstd(means, axis=1)
    std_mean = np.nanmean(stds, axis=1)
    std_mean = np.array([np.diag(i) for i in std_mean])
    std_mean = np.sqrt(std_mean)  # those are actually variances!
    std_var = np.nanvar(stds, axis=1)
    std_var = np.array([np.diag(i) for i in std_var])
    std_relative = std_mean / true_std
    std_std = np.sqrt(std_var)
    # seaborn based plotting
    # create full data table
    data_df = pd.DataFrame()
    for i_model in range(means.shape[2]):
        labels['true_std'] = true_std[:, i_model]
        labels['std_mean'] = std_mean[:, i_model]
        labels['std_var'] = std_var[:, i_model]
        labels['std_relative'] = std_relative[:, i_model]
        labels['std_std'] = std_std[:, i_model]
        labels['model_layer'] = i_model
        data_df = data_df.append(labels)
    data_df = data_df.astype({'n_subj': 'int', 'n_stim': 'int',
                              'n_rep': 'int'})
    with sns.axes_style('ticks'):
        sns.set_context('paper', font_scale=2)
        # change in true Std
        g1 = sns.catplot(data=data_df,
                         x='n_stim', y='true_std', hue='n_subj',
                         kind='point', ci='sd', palette='Blues_d', dodge=.2)
        plt.ylim(bottom=0)
        sns.despine(trim=True, offset=5)
        g2 = sns.catplot(data=data_df,
                         x='n_subj', y='true_std', hue='n_stim',
                         kind='point', ci='sd', palette='Greens_d', dodge=.2)
        plt.ylim(bottom=0)
        sns.despine(trim=True, offset=5)
        g3 = sns.catplot(data=data_df,
                         x='n_rep', y='true_std', hue='n_stim',
                         kind='point', ci='sd', palette='Greens_d', dodge=.2)
        plt.ylim(bottom=0)
        sns.despine(trim=True, offset=5)
        # compare bootstrap to true_std
        # scatterplot
        g4 = sns.FacetGrid(data_df, col='boot_type', aspect=1)
        g4.map(sns.scatterplot, 'true_std', 'std_mean')
        plt.ylim(bottom=0)
        plt.xlim(left=0)
        plt.plot([0, plt.ylim()[1]], [0, plt.ylim()[1]], 'k--')
        sns.despine(trim=True, offset=5)
        # relative deviation of the mean
        g5 = sns.FacetGrid(data_df, col='boot_type')
        g5.map(sns.scatterplot, 'true_std', 'std_relative')
        plt.xlim(left=0)
        plt.plot([0, plt.xlim()[1]], [1, 1], 'k--')
        # relative mean and variance against n
        g6 = sns.catplot(data=data_df, col='boot_type',
                         x='n_stim', y='std_relative', hue='n_subj',
                         kind='point', ci='sd', palette='Blues_d', dodge=.2,
                         order=[10, 20, 40, 80, 160])
        plt.plot([0, plt.xlim()[1]], [1, 1], 'k--')
        sns.despine(trim=True, offset=5)
        g7 = sns.catplot(data=data_df, col='boot_type',
                         x='n_subj', y='std_relative', hue='n_stim',
                         kind='point', ci='sd', palette='Greens_d', dodge=.2)
        plt.plot([0, plt.xlim()[1]], [1, 1], 'k--')
        sns.despine(trim=True, offset=5)
        g8 = sns.catplot(data=data_df, col='boot_type',
                         x='n_rep', y='std_relative', hue='n_stim',
                         kind='point', ci='sd', palette='Greens_d', dodge=.2)
        plt.plot([0, plt.xlim()[1]], [1, 1], 'k--')
        sns.despine(trim=True, offset=5)
        g9 = sns.catplot(data=data_df, col='boot_type',
                         x='n_stim', y='std_std', hue='n_subj',
                         kind='point', ci='sd', palette='Blues_d', dodge=.2)
        sns.despine(trim=True, offset=5)
        g10 = sns.catplot(data=data_df, col='boot_type',
                          x='n_subj', y='std_std', hue='n_stim',
                          kind='point', ci='sd', palette='Greens_d', dodge=.2)
        sns.despine(trim=True, offset=5)
        g11 = sns.catplot(data=data_df, col='boot_type',
                          x='n_rep', y='std_std', hue='n_stim',
                          kind='point', ci='sd', palette='Greens_d', dodge=.2)
        sns.despine(trim=True, offset=5)

        if savefig:
            g1.fig.savefig('figures/true_std_stim_%s.pdf' % variation)
            g2.fig.savefig('figures/true_std_subj_%s.pdf' % variation)
            g3.fig.savefig('figures/true_std_rep_%s.pdf' % variation)
            g4.fig.savefig('figures/std_scatter_%s.pdf' % variation)
            g5.fig.savefig('figures/std_rel_scatter_%s.pdf' % variation)
            g6.fig.savefig('figures/std_rel_stim_%s.pdf' % variation)
            g7.fig.savefig('figures/std_rel_subj_%s.pdf' % variation)
            g8.fig.savefig('figures/std_rel_rep_%s.pdf' % variation)
            g9.fig.savefig('figures/std_std_stim_%s.pdf' % variation)
            g10.fig.savefig('figures/std_std_subj_%s.pdf' % variation)
            g11.fig.savefig('figures/std_std_rep_%s.pdf' % variation)


def plot_eco_paper(simulation_folder='sim_eco', savefig=False):
    labels = pd.read_csv(os.path.join(simulation_folder, 'labels.csv'))
    means = np.load(os.path.join(simulation_folder, 'means.npy'))
    stds = np.load(os.path.join(simulation_folder, 'stds.npy'))
    means = means[:len(labels)]
    # remove nan entries
    idx_nan = ~np.any(np.isnan(means[:, :, 0]), axis=1)
    labels = labels[list(idx_nan)]
    means = means[idx_nan]
    stds = stds[idx_nan]
    # compute statistics
    true_std = np.nanstd(means, axis=1)
    std_mean = np.nanmean(stds, axis=1)
    std_mean = np.array([np.diag(i) for i in std_mean])
    std_mean = np.sqrt(std_mean)  # those are actually variances!
    std_var = np.nanvar(stds, axis=1)
    std_var = np.array([np.diag(i) for i in std_var])
    std_relative = std_mean / true_std
    std_std = np.sqrt(std_var)
    snr = (np.var(np.mean(means, axis=1), axis=1)
           / np.mean(np.var(means, axis=1), axis=1))
    # seaborn based plotting
    # create full data table
    data_df = pd.DataFrame()
    for i_model in range(means.shape[2]):
        labels['true_std'] = true_std[:, i_model]
        labels['std_mean'] = std_mean[:, i_model]
        labels['std_var'] = std_var[:, i_model]
        labels['std_relative'] = std_relative[:, i_model]
        labels['std_std'] = std_std[:, i_model]
        labels['model_layer'] = i_model
        labels['snr'] = snr
        labels['log-snr'] = np.log10(snr)
        data_df = data_df.append(labels)
    data_df = data_df.astype({'n_subj': 'int', 'n_stim': 'int',
                              'n_rep': 'int', 'sigma_noise': 'int'})
    data_df.loc[pd.isna(data_df['variation']), 'variation'] = 'none'
    # for 100 observations we expect this much std even if we were perfect:
    std_expected = np.std(np.sqrt(scipy.stats.f(1000, 100).rvs(100000)))
    # this is the area marked by the gray rectangle in the plot
    CI = [1 - std_expected, 1 + std_expected]
    with sns.axes_style('ticks'):
        sns.set_context('paper', font_scale=2)
        #### change in SNR ####
        # appendix version
        g1 = sns.catplot(data=data_df, legend=False, col='variation',
                         row='sigma_noise',
                         x='n_stim', y='log-snr', hue='n_subj',
                         kind='point', ci='sd', palette='Blues_d', dodge=.2)
        g1.add_legend(
            frameon=False, title='# of rdms',
            bbox_to_anchor=(1.0, 1.0), loc=2)
        g1.set_xlabels('# of stimuli')
        g1.set_ylabels('signal to noise ratio')
        plt.ylim([-2, 3.5])
        plt.yticks([-2, -1, 0, 1, 2, 3],
                   ['10^-2', '10^-1', '10^0', '10^1', '10^2', '10^3'])
        sns.despine(trim=True, offset=5)

        g2 = sns.catplot(data=data_df, legend=False, col='variation',
                         row='sigma_noise',
                         x='n_subj', y='log-snr', hue='sd',
                         kind='point', ci='sd', palette='Greens_d', dodge=.2)
        g2.add_legend(
            frameon=False, title='# of stimuli',
            bbox_to_anchor=(1.0, 1.0), loc=2)
        g2.set_xlabels('# of subjects')
        g2.set_ylabels('signal to noise ratio')
        plt.ylim([-2, 3.5])
        plt.yticks([-2, -1, 0, 1, 2, 3],
                   ['10^-2', '10^-1', '10^0', '10^1', '10^2', '10^3'])
        sns.despine(trim=True, offset=5)

        g3 = sns.catplot(data=data_df, legend=False, col='variation',
                         row='sigma_noise',
                         x='n_rep', y='log-snr', hue='n_subj',
                         kind='point', ci='sd', palette='Greens_d', dodge=.2)
        g3.add_legend(
            frameon=False, title='# of stimuli',
            bbox_to_anchor=(1.0, 1.0), loc=2)
        g3.set_xlabels('# of repetitions')
        g3.set_ylabels('signal to noise ratio')
        plt.ylim([-2, 3.5])
        plt.yticks([-2, -1, 0, 1, 2, 3],
                   ['10^-2', '10^-1', '10^0', '10^1', '10^2', '10^3'])
        sns.despine(trim=True, offset=5)

        # main text version
        g1_m = sns.catplot(data=data_df, legend=False,
                           x='n_stim', y='log-snr', hue='n_subj',
                           kind='point', ci='sd', palette='Blues_d', dodge=.2)
        g1_m.add_legend(
            frameon=False, title='# of subjects',
            bbox_to_anchor=(1.0, 1.0), loc=2)
        g1_m.set_xlabels('# of stimuli')
        g1_m.set_ylabels('signal to noise ratio')
        plt.ylim([-2, 3.5])
        plt.yticks([-2, -1, 0, 1, 2, 3],
                   ['10^-2', '10^-1', '10^0', '10^1', '10^2', '10^3'])
        sns.despine(trim=True, offset=5)

        g2_m = sns.catplot(data=data_df, legend=False,
                           x='n_subj', y='log-snr', hue='n_stim',
                           kind='point', ci='sd', palette='Greens_d', dodge=.2)
        g2_m.add_legend(
            frameon=False, title='# of stimuli',
            bbox_to_anchor=(1.0, 1.0), loc=2)
        g2_m.set_xlabels('# of subjects')
        g2_m.set_ylabels('signal to noise ratio')
        plt.ylim([-2, 3.5])
        plt.yticks([-2, -1, 0, 1, 2, 3],
                   ['10^-2', '10^-1', '10^0', '10^1', '10^2', '10^3'])
        sns.despine(trim=True, offset=5)

        g3_m = sns.catplot(data=data_df, legend=False,
                           x='n_subj', y='log-snr', hue='n_rep',
                           kind='point', ci='sd', palette='Reds_d', dodge=.2)
        g3_m.add_legend(
            frameon=False, title='# of repetitions',
            bbox_to_anchor=(1.0, 1.0), loc=2)
        g3_m.set_xlabels('# of subjects')
        g3_m.set_ylabels('signal to noise ratio')
        plt.ylim([-2, 3.5])
        plt.yticks([-2, -1, 0, 1, 2, 3],
                   ['10^-2', '10^-1', '10^0', '10^1', '10^2', '10^3'])
        sns.despine(trim=True, offset=5)

        g9_m = sns.catplot(data=data_df, legend=False,
                           x='sd', y='log-snr', hue='sigma_noise',
                           kind='point', ci='sd', palette='Greys', dodge=.2)
        g9_m.add_legend(
            frameon=False, title='noise std',
            bbox_to_anchor=(1.0, 1.0), loc=2)
        g9_m.set_ylabels('signal to noise ratio')
        g9_m.set_xlabels('voxel size [prop. of image]')
        plt.ylim([-3, 3.5])
        plt.yticks([-2, -1, 0, 1, 2, 3],
                   ['10^-2', '10^-1', '10^0', '10^1', '10^2', '10^3'])
        sns.despine(trim=True, offset=5)

        g10_m = sns.catplot(data=data_df, legend=False,
                            x='variation', y='log-snr', hue='sigma_noise',
                            kind='point', ci='sd', palette='Greys', dodge=.2,
                            order=['none', 'subj', 'stim', 'both'])
        g10_m.add_legend(
            frameon=False, title='noise std',
            bbox_to_anchor=(1.0, 1.0), loc=2)
        g10_m.set_xlabels('variation')
        g10_m.set_ylabels('signal to noise ratio')
        plt.ylim([-2, 3.5])
        plt.yticks([-2, -1, 0, 1, 2, 3],
                   ['10^-2', '10^-1', '10^0', '10^1', '10^2', '10^3'])
        plt.xticks(
            [0, 1, 2, 3],
            labels=['none', 'subjects', 'stimuli', 'both'])
        sns.despine(trim=True, offset=5)

        # compare bootstrap to true_std
        # make subsets based on variation
        dat_none = data_df[data_df['variation'] == 'none']
        dat_none_rdm = dat_none[dat_none['boot_type'] == 'rdm']
        dat_none_pat = dat_none[dat_none['boot_type'] == 'pattern']
        dat_subj = data_df[data_df['variation'] == 'subj']
        dat_subj = dat_subj[dat_subj['boot_type'] == 'rdm']
        dat_stim = data_df[data_df['variation'] == 'stim']
        dat_stim = dat_stim[dat_stim['boot_type'] == 'pattern']
        dat_both = data_df[data_df['variation'] == 'both']
        dat_both = dat_both[(dat_both['boot_type'] == 'both')
                            | (dat_both['boot_type'] == 'fancyboot')]

        # relative standard deviation
        g4_m = sns.catplot(data=dat_none_rdm, legend=False,
                           x='n_subj', y='std_relative', hue='n_stim',
                           kind='point', ci='sd', palette='Greens_d', dodge=.2,
                           order=[5, 10, 20, 40, 80])
        plt.plot([-0.3, 4.3], [1, 1], 'k--')
        r = plt.Rectangle([-0.3, CI[0]], 4.6, CI[1]-CI[0],
                          facecolor='gray', zorder=-1, alpha=0.5)
        g4_m.ax.add_patch(r)
        sns.despine(trim=True, offset=5)
        plt.title('RDM-bootstrap, no true variation')
        g4_m.add_legend(
            frameon=False, title='# of stimuli',
            bbox_to_anchor=(1.0, 1.0), loc=2)
        g4_m.set_xlabels('# of subjects')
        g4_m.set_ylabels(r'relative uncertainty $[\sigma_{boot}/\sigma_{true}]$')

        g5_m = sns.catplot(data=dat_none_pat, legend=False,
                           x='n_stim', y='std_relative', hue='n_subj',
                           kind='point', ci='sd', palette='Blues_d', dodge=.2,
                           order=[10, 20, 40, 80, 160])
        plt.plot([-0.3, 4.3], [1, 1], 'k--')
        r = plt.Rectangle([-0.3, CI[0]], 4.6, CI[1]-CI[0],
                          facecolor='gray', zorder=-1, alpha=0.5)
        g5_m.ax.add_patch(r)
        sns.despine(trim=True, offset=5)
        plt.title('pattern-bootstrap, no true variation')
        g5_m.add_legend(
            frameon=False, title='# of subjects',
            bbox_to_anchor=(1.0, 1.0), loc=2)
        g5_m.set_xlabels('# of stimuli')
        g5_m.set_ylabels(r'relative uncertainty $[\sigma_{boot}/\sigma_{true}]$')

        g6_m = sns.catplot(data=dat_stim, legend=False,
                           x='n_stim', y='std_relative', hue='n_subj',
                           kind='point', ci='sd', palette='Blues_d', dodge=.2,
                           order=[10, 20, 40, 80, 160])
        plt.plot([-0.3, 4.3], [1, 1], 'k--')
        r = plt.Rectangle([-0.3, CI[0]], 4.6, CI[1]-CI[0],
                          facecolor='gray', zorder=-1, alpha=0.5)
        g6_m.ax.add_patch(r)
        sns.despine(trim=True, offset=5)
        plt.title('pattern-bootstrap, stimulus variation')
        g6_m.add_legend(
            frameon=False, title='# of subjects',
            bbox_to_anchor=(1.0, 1.0), loc=2)
        g6_m.set_xlabels('# of stimuli')
        g6_m.set_ylabels(r'relative uncertainty $[\sigma_{boot}/\sigma_{true}]$')

        g7_m = sns.catplot(data=dat_subj, legend=False,
                           x='n_subj', y='std_relative', hue='n_stim',
                           kind='point', ci='sd', palette='Greens_d', dodge=.2,
                           order=[5, 10, 20, 40, 80])
        plt.plot([-0.3, 4.3], [1, 1], 'k--')
        r = plt.Rectangle([-0.3, CI[0]], 4.6, CI[1]-CI[0],
                          facecolor='gray', zorder=-1, alpha=0.5)
        g7_m.ax.add_patch(r)
        sns.despine(trim=True, offset=5)
        plt.title('RDM-bootstrap, subject variation')
        g7_m.add_legend(
            frameon=False, title='# of subjects',
            bbox_to_anchor=(1.0, 1.0), loc=2)
        g7_m.set_xlabels('# of stimuli')
        g7_m.set_ylabels(r'relative uncertainty $[\sigma_{boot}/\sigma_{true}]$')

        g8_m = sns.catplot(data=dat_both, col='boot_type', legend=False,
                           x='n_stim', y='std_relative', hue='n_subj',
                           kind='point', ci='sd', palette='Blues', dodge=.2,
                           order=[10, 20, 40, 80, 160])
        g8_m.axes[0, 0].plot([-0.3, 4.3], [1, 1], 'k--')
        g8_m.axes[0, 1].plot([-0.3, 4.3], [1, 1], 'k--')
        r = plt.Rectangle([-0.3, CI[0]], 4.6, CI[1]-CI[0],
                          facecolor='gray', zorder=-1, alpha=0.5)
        g8_m.axes[0, 0].add_patch(r)
        r = plt.Rectangle([-0.3, CI[0]], 4.6, CI[1]-CI[0],
                          facecolor='gray', zorder=-1, alpha=0.5)
        g8_m.axes[0, 1].add_patch(r)
        sns.despine(trim=True, offset=5)
        g8_m.axes[0, 0].set_title('double-bootstrap, both varied')
        g8_m.axes[0, 1].set_title('dual bootstrap, both varied')
        g8_m.add_legend(
            frameon=False, title='# of subjects',
            bbox_to_anchor=(1.0, 1.0), loc=2)
        g8_m.set_xlabels('# of stimuli')
        g8_m.set_ylabels(r'relative uncertainty $[\sigma_{boot}/\sigma_{true}]$')

        if savefig:
            g1_m.fig.savefig('figures/SNR_stim.pdf', bbox_inches='tight')
            g2_m.fig.savefig('figures/SNR_subj.pdf', bbox_inches='tight')
            g3_m.fig.savefig('figures/SNR_rep.pdf', bbox_inches='tight')
            g4_m.fig.savefig('figures/std_rel_none_rdm.pdf', bbox_inches='tight')
            g5_m.fig.savefig('figures/std_rel_none_pattern.pdf', bbox_inches='tight')
            g6_m.fig.savefig('figures/std_rel_stim_pattern.pdf', bbox_inches='tight')
            g7_m.fig.savefig('figures/std_rel_subj_rdm.pdf', bbox_inches='tight')
            g8_m.fig.savefig('figures/std_rel_both.pdf', bbox_inches='tight')
            g9_m.fig.savefig('figures/SNR_vox_size.pdf', bbox_inches='tight')
            g10_m.fig.savefig('figures/SNR_variation.pdf', bbox_inches='tight')


def plot_allen(result_file='allen_results.npz', task_file='allen_tasks.csv'):
    labels = pd.read_csv(task_file, index_col=0)
    results = np.load(result_file)
    means = results['means']
    variances = results['variances']
    true_var = np.var(means, 1)
    true_std = np.sqrt(true_var)
    cov_estimate = np.mean(variances, 1)
    var_estimate = np.einsum('ijj->ij', cov_estimate)
    std_relative = np.sqrt(var_estimate / true_var)
    snr = (np.var(np.mean(means, axis=1), axis=1)
           / np.mean(np.var(means, axis=1), axis=1))
    # seaborn based plotting
    # create full data table
    data_df = pd.DataFrame()
    for i_model in range(means.shape[2]):
        labels['true_std'] = true_std[:, i_model]
        labels['var_estimate'] = var_estimate[:, i_model]
        labels['std_relative'] = std_relative[:, i_model]
        labels['model'] = i_model
        labels['snr'] = snr
        labels['log-snr'] = np.log10(snr)
        data_df = data_df.append(labels)
    data_df = data_df.astype({'n_subj': 'int', 'n_stim': 'int',
                              'n_repeat': 'int', 'n_cell': 'int'})

    # for 100 observations we expect this much std even if we were perfect:
    std_expected = np.std(np.sqrt(scipy.stats.f(1000, 100).rvs(100000)))
    # this is the area marked by the gray rectangle in the plot
    CI = [1 - std_expected, 1 + std_expected]

    g1_m = sns.catplot(data=data_df, legend=False,
                       x='n_subj', y='std_relative', hue='n_stim',
                       kind='point', ci='sd', palette='Greens_d', dodge=.2,
                       order=[5, 10, 15])
    plt.plot([-0.3, 2.3], [1, 1], 'k--')
    r = plt.Rectangle([-0.3, CI[0]], 2.6, CI[1]-CI[0],
                      facecolor='gray', zorder=-1, alpha=0.5)
    g1_m.ax.add_patch(r)
    sns.despine(trim=True, offset=5)
    plt.title('double bootstrap')
    g1_m.add_legend(
        frameon=False, title='# of stimuli',
        bbox_to_anchor=(1.0, 1.0), loc=2)
    g1_m.set_xlabels('# of subjects')
    g1_m.set_ylabels(r'relative uncertainty $[\sigma_{boot}/\sigma_{true}]$')

    g2_m = sns.catplot(data=data_df, col='rdm_comparison', legend=False,
                       x='n_stim', y='std_relative', hue='n_subj',
                       kind='point', ci='sd', palette='Blues', dodge=.2,
                       order=[10, 20, 40])
    for ax in g2_m.axes[0]:
        ax.plot([-0.3, 2.3], [1, 1], 'k--')
        r = plt.Rectangle([-0.3, CI[0]], 2.6, CI[1]-CI[0],
                          facecolor='gray', zorder=-1, alpha=0.5)
        ax.add_patch(r)
    sns.despine(trim=True, offset=5)
    g2_m.axes[0, 0].set_title('double-bootstrap')
    g2_m.axes[0, 1].set_title('dual bootstrap')
    g2_m.add_legend(
        frameon=False, title='# of subjects',
        bbox_to_anchor=(1.0, 1.0), loc=2)
    g2_m.set_xlabels('# of stimuli')
    g2_m.set_ylabels(r'relative uncertainty $[\sigma_{boot}/\sigma_{true}]$')

    g3_m = sns.catplot(data=data_df, legend=False,
                       x='n_stim', y='log-snr', hue='n_subj',
                       kind='point', ci='sd', palette='Greens_d', dodge=.2,
                       order=[10, 20, 40])
    g3_m.set_xlabels('# of stimuli')
    g3_m.set_ylabels('signal to noise ratio')
    plt.ylim([-3, 1])
    plt.yticks([-3, -2, -1, 0, 1],
               ['10^-3', '10^-2', '10^-1', '10^0', '10^1'])
    sns.despine(trim=True, offset=5)
    g3_m.add_legend(
        frameon=False, title='# of subjects',
        bbox_to_anchor=(1.0, 1.0), loc=2)

    g4_m = sns.catplot(data=data_df, legend=False,
                       x='n_cell', y='log-snr', hue='n_repeat',
                       kind='point', ci='sd', palette='Reds_d', dodge=.2,
                       order=[10, 20, 40])
    g4_m.set_xlabels('# of cells per subject')
    g4_m.set_ylabels('signal to noise ratio')
    plt.ylim([-3, 1])
    plt.yticks([-3, -2, -1, 0, 1],
               ['10^-3', '10^-2', '10^-1', '10^0', '10^1'])
    sns.despine(trim=True, offset=5)
    g4_m.add_legend(
        frameon=False, title='# of repeats',
        bbox_to_anchor=(1.0, 1.0), loc=2)

    g5_m = sns.catplot(data=data_df, legend=False,
                       x='rdm_comparison', y='log-snr', hue='n_repeat',
                       kind='point', ci='sd', palette='Reds_d', dodge=.2,
                       order=['corr', 'corr_cov', 'cosine', 'cosine_cov'])
    g5_m.set_xlabels('RDM comparison method')
    g5_m.set_ylabels('signal to noise ratio')
    plt.ylim([-3, 1])
    plt.yticks([-3, -2, -1, 0, 1],
               ['10^-3', '10^-2', '10^-1', '10^0', '10^1'])
    sns.despine(trim=True, offset=5)
    g5_m.add_legend(
        frameon=False, title='# of repeats',
        bbox_to_anchor=(1.0, 1.0), loc=2)

    g6_m = sns.catplot(data=data_df, legend=False,
                       x='targeted_structure', y='log-snr', hue='n_repeat',
                       kind='point', ci='sd', palette='Reds_d', dodge=.2,
                       order=['corr', 'corr_cov', 'cosine', 'cosine_cov'])
    g6_m.set_xlabels('# of cells per subject')
    g5_m.set_ylabels('signal to noise ratio')
    plt.ylim([-3, 1])
    plt.yticks([-3, -2, -1, 0, 1],
               ['10^-3', '10^-2', '10^-1', '10^0', '10^1'])
    sns.despine(trim=True, offset=5)
    g5_m.add_legend(
        frameon=False, title='# of repeats',
        bbox_to_anchor=(1.0, 1.0), loc=2)


def plot_metrics(simulation_folder='sim_metric', savefig=False):
    labels = pd.read_csv(os.path.join(simulation_folder, 'labels.csv'))
    means = np.load(os.path.join(simulation_folder, 'means.npy'))
    stds = np.load(os.path.join(simulation_folder, 'stds.npy'))
    means = means[:len(labels)]
    # remove nan entries
    idx_nan = ~np.any(np.isnan(means[:, :, 0]), axis=1)
    labels = labels[list(idx_nan)]
    means = means[idx_nan]
    stds = stds[idx_nan]
    # compute statistics
    true_std = np.nanstd(means, axis=1)
    std_mean = np.nanmean(stds, axis=1)
    std_mean = np.array([np.diag(i) for i in std_mean])
    std_mean = np.sqrt(std_mean)  # those are actually variances!
    std_var = np.nanvar(stds, axis=1)
    std_var = np.array([np.diag(i) for i in std_var])
    std_relative = std_mean / true_std
    std_std = np.sqrt(std_var)
    snr = (np.var(np.mean(means, axis=1), axis=1)
           / np.mean(np.var(means, axis=1), axis=1))
    # seaborn based plotting
    # create full data table
    data_df = pd.DataFrame()
    for i_model in range(means.shape[2]):
        labels['true_std'] = true_std[:, i_model]
        labels['std_mean'] = std_mean[:, i_model]
        labels['std_var'] = std_var[:, i_model]
        labels['std_relative'] = std_relative[:, i_model]
        labels['std_std'] = std_std[:, i_model]
        labels['model_layer'] = i_model
        labels['snr'] = snr
        labels['log-snr'] = np.log10(snr)
        data_df = data_df.append(labels)
    data_df = data_df.astype({'n_subj': 'int', 'n_stim': 'int',
                              'n_rep': 'int', 'sigma_noise': 'int'})
    data_df.loc[pd.isna(data_df['variation']), 'variation'] = 'none'
    sns.set_context('paper', font_scale=1.5)
    sns.set_style("ticks", {"xtick.major.size": 8, "ytick.major.size": 8})
    g1 = sns.catplot(data=data_df, legend=False,
                     x='rdm_comparison', y='log-snr', hue='variation',
                     kind='point', ci='sd', dodge=0, aspect=0.6,
                     order=['kendall', 'tau-a', 'spearman', 'rho-a', '',
                            'corr', 'cosine', 'corr_cov', 'cosine_cov'],
                     hue_order=['none', 'subj', 'stim', 'both'],
                     hue_labels=['none', 'subjects', 'stimuli', 'both'])
    g1.add_legend(
        frameon=False, title='varied dimensions',
        bbox_to_anchor=(1.0, 1.0), loc=2)
    g1.set_xlabels('rdm comparison method')
    g1.set_ylabels('signal to noise ratio')
    plt.ylim([0, 3])
    plt.yticks([0, 1, 2, 3],
               ['10^0', '10^1', '10^2', '10^3'])
    sns.despine(trim=True, offset=5)
    plt.xticks([0, 1, 2, 3, 5, 6, 7, 8],
               ['tau-a', 'tau-b', 'rho-a', 'rho-b',
                'corr', 'cosine', 'whitened corr', 'whitened cosine'],
               rotation=90)
    if savefig:
        g1.fig.savefig('figures/metrics.pdf', bbox_inches='tight')


def plot_flex(simulation_folder='sim_flex', savefig=False):
    labels = pd.read_csv(os.path.join(simulation_folder, 'labels.csv'))
    means = np.load(os.path.join(simulation_folder, 'means.npy'))
    stds = np.load(os.path.join(simulation_folder, 'stds.npy'))
    labels['full_model'] = labels['model_type'].astype(str) \
        + '_' + labels['sd_fit'].astype(str)
    means = means[:len(labels)]
    # remove nan entries
    idx_nan = ~np.any(np.isnan(means[:, :, 0]), axis=1)
    labels = labels[list(idx_nan)]
    means = means[idx_nan]
    stds = stds[idx_nan]
    # compute statistics
    true_std = np.nanstd(means, axis=1)
    std_mean = np.nanmean(stds, axis=1)
    std_mean = np.array([np.diag(i) for i in std_mean])
    std_mean = np.sqrt(std_mean)  # those are actually variances!
    std_var = np.nanvar(stds, axis=1)
    std_var = np.array([np.diag(i) for i in std_var])
    std_relative = std_mean / true_std
    std_std = np.sqrt(std_var)
    snr = (np.var(np.mean(means, axis=1), axis=1)
           / np.mean(np.var(means, axis=1), axis=1))
    means_other = np.concatenate((means[:, :, :8], means[:, :, 9:]), axis=2)
    true_best = np.mean(means[:, :, 8], 1) \
        - np.max(np.mean(means_other[:, :, :], 1), 1)
    # seaborn based plotting
    # create full data table
    data_df = pd.DataFrame()
    for i_model in range(means.shape[2]):
        labels['true_std'] = true_std[:, i_model]
        labels['std_mean'] = std_mean[:, i_model]
        labels['std_var'] = std_var[:, i_model]
        labels['std_relative'] = std_relative[:, i_model]
        labels['std_std'] = std_std[:, i_model]
        labels['model_layer'] = i_model
        labels['snr'] = snr
        labels['log-snr'] = np.log10(snr)
        labels['true_perf'] = np.mean(means[:, :, 8], 1)
        labels['true-best'] = true_best
        data_df = data_df.append(labels)
    data_df = data_df.astype({'n_subj': 'int', 'n_stim': 'int',
                              'n_rep': 'int', 'sigma_noise': 'int'})
    data_df.loc[pd.isna(data_df['variation']), 'variation'] = 'none'
    data_df.loc[data_df['model_type'] == 'select_full', 'sd_fit'] = 'fit'
    data_df.loc[data_df['model_type'] == 'select_average', 'sd_fit'] = 'fit'
    data_df.loc[data_df['model_type'] == 'select_both', 'sd_fit'] = 'fit'
    data_df.loc[data_df['model_type'] == 'select_mean', 'sd_fit'] = 'fit'
    data_df.loc[data_df['model_type'] == 'weighted_avgfull', 'sd_fit'] = 'fit'
    sns.set_context('paper', font_scale=1.5)
    sns.set_style("ticks", {"xtick.major.size": 8, "ytick.major.size": 8})
    sns.color_palette('muted')
    g1 = sns.catplot(data=data_df, legend=False,
                     x='model_type', y='snr', hue='sd_fit',
                     kind='point', ci='sd', dodge=0,
                     order=['fixed_full', 'fixed_average', 'fixed_mean', '',
                            'select_full', 'select_average', 'select_both',
                            'select_mean', '',
                            'weighted_avgfull'],
                     hue_order=[0.0, 0.05, np.inf, 'fit'],
                     palette=['r', 'b', 'y', 'k'])
    g1.add_legend(
        frameon=False, title='assumed voxel size',
        bbox_to_anchor=(1.0, 1.0), loc=2)
    g1.set_xlabels('model type')
    g1.set_ylabels('signal to noise ratio')
    sns.despine(trim=True, offset=5)
    plt.xticks(
        [0, 1, 2, 4, 5, 6, 7, 9],
        ['fixed: features',
         'fixed: average',
         'fixed: weighted',
         'selection: features',
         'selection: average',
         'selection: both',
         'selection: weighted',
         'linear combination'], rotation=90)
    g2 = sns.catplot(data=data_df, legend=False,
                     x='model_type', y='true_perf', hue='sd_fit',
                     kind='point', ci='sd', dodge=0,
                     order=['fixed_full', 'fixed_average', 'fixed_mean', '',
                            'select_full', 'select_average', 'select_both',
                            'select_mean', '',
                            'weighted_avgfull'],
                     hue_order=[0.0, 0.05, np.inf, 'fit'],
                     palette=['r', 'b', 'y', 'k'])
    g2.add_legend(
        frameon=False, title='assumed voxel size',
        bbox_to_anchor=(1.0, 1.0), loc=2)
    g2.set_xlabels('model type')
    g2.set_ylabels('average corr. of true layer')
    sns.despine(trim=True, offset=5)
    plt.xticks(
        [0, 1, 2, 4, 5, 6, 7, 9],
        ['fixed: features',
         'fixed: average',
         'fixed: weighted',
         'selection: features',
         'selection: average',
         'selection: both',
         'selection: weighted',
         'linear combination'], rotation=90)
    g3 = sns.catplot(data=data_df, legend=False,
                     x='model_type', y='true-best', hue='sd_fit',
                     kind='point', ci='sd', dodge=0,
                     order=['fixed_full', 'fixed_average', 'fixed_mean', '',
                            'select_full', 'select_average', 'select_both',
                            'select_mean', '',
                            'weighted_avgfull'],
                     hue_order=[0.0, 0.05, np.inf, 'fit'],
                     palette=['r', 'b', 'y', 'k'])
    g3.add_legend(
        frameon=False, title='assumed voxel size',
        bbox_to_anchor=(1.0, 1.0), loc=2)
    g3.set_xlabels('model type')
    g3.set_ylabels('true layer - best layer')
    sns.despine(trim=True, offset=5)
    plt.xticks(
        [0, 1, 2, 4, 5, 6, 7, 9],
        ['fixed: features',
         'fixed: average',
         'fixed: weighted',
         'selection: features',
         'selection: average',
         'selection: both',
         'selection: weighted',
         'linear combination'], rotation=90)
    plt.plot([-0, 9], [0, 0], 'k--')
    g3.ax.set_position(g2.ax.get_position())

    g4 = sns.catplot(data=data_df, legend=False,
                     x='model_type', y='std_relative', hue='sd_fit',
                     kind='swarm', ci='sd', dodge=0,
                     order=['fixed_full', 'fixed_average', 'fixed_mean', '',
                            'select_full', 'select_average', 'select_both',
                            'select_mean', '',
                            'weighted_avgfull'],
                     hue_order=[0.0, 0.05, np.inf, 'fit'],
                     palette=['r', 'b', 'y', 'k'])
    g4.add_legend(
        frameon=False, title='assumed voxel size',
        bbox_to_anchor=(1.0, 1.0), loc=2)
    g4.set_xlabels('model type')
    g4.set_ylabels('relative uncertainty')
    sns.despine(trim=True, offset=5)
    plt.xticks(
        [0, 1, 2, 4, 5, 6, 7, 9],
        ['fixed: features',
         'fixed: average',
         'fixed: weighted',
         'selection: features',
         'selection: average',
         'selection: both',
         'selection: weighted',
         'linear combination'], rotation=90)
    plt.plot([-0.5, 9.5], [1, 1], 'k--')

    plt.figure()
    g5 = sns.histplot(data=data_df, stat="count", multiple="stack",
                      x='std_relative', hue='model_type')
    plt.xticks([0.8, 0.9, 1, 1.1, 1.2, 1.3])
    plt.xlabel('Relative Uncertainty')
    sns.despine(trim=True, offset=5)
    g5.legend(
        ['f:f',
         'f:a',
         'f:w',
         's:f',
         's:a',
         's:b',
         's:w',
         'lin'],
        frameon=False, title='Model Type',
        bbox_to_anchor=(1.0, 0.5), loc=6)

    if savefig:
        g1.fig.savefig('figures/flex_snr.pdf', bbox_inches='tight')
        g2.fig.savefig('figures/flex_true_perf.pdf', bbox_inches='tight')
        g3.fig.savefig('figures/flex_best.pdf', bbox_inches='tight')
        g4.fig.savefig('figures/flex_calibration.pdf', bbox_inches='tight')
        g5.figure.savefig('figures/flex_hist.pdf', bbox_inches='tight')


def plot_boot_cv(simulation_folder='boot_cv', savefig=False):
    """ plots the bootstrap cv validation """
    # preprocessing
    variances = np.load(os.path.join(simulation_folder, 'var.npy'))
    variances_raw = np.load(os.path.join(simulation_folder, 'var_raw.npy'))
    variances[0] = np.nan
    normalizer = np.nanmean(np.nanmean(variances, 0, keepdims=True),
                            -1, keepdims=True)
    var_norm = variances / normalizer
    var_raw_norm = variances_raw / normalizer

    # for plotting efficiency
    var_normdiag = np.einsum('ijjkl->ijkl', var_norm)
    var_var = np.var(var_normdiag, -1).reshape(10, -1)

    # for plotting accuracy
    var_norm_plot = np.einsum('ijjk->ijk', np.nanmean(var_norm, -1)
                              ).reshape(10, -1)
    var_raw_plot = np.einsum('ijjk->ijk', np.nanmean(var_raw_norm, -1)
                             ).reshape(10, -1)
    x = np.repeat(np.arange(6), 120).reshape(6, -1) \
        + 0.2 * np.random.rand(6, 120) - 0.1
    # Plotting
    # variance estimation accurate
    f_acc = plt.figure()
    with sns.axes_style("ticks"):
        sns.set_context(rc={
            "axes.linewidth": 2.5, "xtick.major.width": 2.5,
            "ytick.major.width": 2.5,
            "xtick.labelsize": 14, "ytick.labelsize": 14,
            "xtick.major.size": 8, "ytick.major.size": 8})
        line_raw = plt.plot(x, var_raw_plot[:6], 'c.',
                            markersize=5, linewidth=2)
        line_norm = plt.plot(x, var_norm_plot[:6], 'k.',
                             markersize=5, linewidth=2)
        plt.plot([-0.5, 5.5], [1, 1], 'k--')
        plt.xticks(np.arange(6), [1, 2, 4, 8, 16, 32], fontsize=16)
        plt.yticks([1, 1.5, 2, 2.5], [1, 1.5, 2, 2.5], fontsize=16)
        plt.box('off')
        sns.despine(trim=True, offset=5)
        plt.xticks(np.arange(6), [1, 2, 4, 8, 16, 32], fontsize=16)
        plt.yticks([1, 1.5, 2, 2.5], [1, 1.5, 2, 2.5], fontsize=16)
        plt.ylabel('normalized variance', fontsize=18)
        plt.xlabel('number of cv-assignments', fontsize=18)
        plt.legend([line_raw[0], line_norm[0]],
                   ['uncorrected', 'corrected'],
                   frameon=False, fontsize=16, markerscale=2)
    # more efficient
    f_eff = plt.figure()
    with sns.axes_style("ticks"):
        sns.set_context(rc={
            "axes.linewidth": 2.5, "xtick.major.width": 2.5,
            "ytick.major.width": 2.5,
            "xtick.labelsize": 14, "ytick.labelsize": 14,
            "xtick.major.size": 8, "ytick.major.size": 8})
        l_line = plt.plot([0.5, 5.5], np.log10([0.0064, 0.0001]), 'k--')
        p_boot = plt.plot(x[2:], np.log10(var_var[6:]), 'k.')
        p_dat = plt.plot(x, np.log10(var_var[:6]), '.', color='grey')
        plt.box('off')
        plt.xticks(np.arange(5) + 1, ['2000', '4000', '8000', '16000', '32000'])
        plt.yticks(np.log10([0.0001, 0.001, 0.01]), [0.0001, 0.001, 0.01])
        sns.despine(trim=True, offset=5)
        plt.ylabel('variance of estimate', fontsize=18)
        plt.xlabel('number of crossvalidations', fontsize=18)
        plt.legend([p_dat[0], p_boot[0], l_line[0]],
                   ['increasing cv', 'increasing boot',
                    'expectation \nmore samples'],
                   frameon=False, fontsize=14, markerscale=2)
    if savefig:
        f_acc.savefig('figures/boot_cv_acc.pdf', bbox_inches='tight')
        f_eff.savefig('figures/boot_cv_eff.pdf', bbox_inches='tight')


def make_illustrations():
    """
    generates and saves a couple of simulated results for illustrations
    3 subjects x 20 measurements x 10 stimuli

    response matrices
    RDMs

    This was generated by stripping down sim_eco

    """
    folder_out = 'illustrations'
    if not os.path.exists(folder_out):
        os.mkdir(folder_out)
    ecoset_path = '~/ecoset/val'
    model = dnn.get_default_model()
    ecoset_path = pathlib.Path(ecoset_path).expanduser()
    sd = 0.05
    n_stim_all = 10
    n_stim = 10
    layer = 8
    n_subj = 3
    n_voxel = 20
    n_repeat = 1
    duration = 1
    pause = 1
    endzeros = 25
    use_cor_noise = True
    sigma_noise = 1
    resolution = 2
    ar_coeff = 0.5
    rdm_type = 'mahalanobis'
    noise_type = 'eye'
    # get stimulus list or save one if there is none yet
    stim_paths = []
    folders = os.listdir(ecoset_path)
    folders = [f for f in folders if f[0] != '.']
    for i_stim in range(n_stim_all):
        i_folder = np.random.randint(len(folders))
        images = os.listdir(os.path.join(ecoset_path,
                                         folders[i_folder]))
        images = [i for i in images if i[0] != '.']
        i_image = np.random.randint(len(images))
        stim_paths.append(os.path.join(folders[i_folder],
                                       images[i_image]))
    stim_list = get_stimuli_ecoset(ecoset_path, stim_paths[:n_stim])
    # Recalculate U_complete if necessary
    U_complete = []
    for i_stim in range(n_stim):
        U_complete.append(
            dnn.get_complete_representation(
                model=model, layer=layer,
                stimulus=stim_list[i_stim]))
    U_shape = np.array(U_complete[0].shape)

    # get new sampling locations if necessary
    sigmaP = []
    indices_space = []
    weights = []
    for i_subj in range(n_subj):
        indices_space_subj, weights_subj = dnn.get_random_indices_conv(
            U_shape, n_voxel)
        sigmaP_subj = dnn.get_sampled_sigmaP(
            U_shape, indices_space_subj, weights_subj, [sd, sd])
        sigmaP.append(sigmaP_subj)
        indices_space.append(indices_space_subj)
        weights.append(weights_subj)
    sigmaP = np.array(sigmaP)
    indices_space = np.array(indices_space)
    weights = np.array(weights)
    # extract new dnn activations
    Utrue = []
    for i_subj in range(n_subj):
        Utrue_subj = [dnn.sample_representation(np.squeeze(U_c),
                                                indices_space[i_subj],
                                                weights[i_subj],
                                                [sd, sd])
                      for U_c in U_complete]
        Utrue_subj = np.array(Utrue_subj)
        Utrue_subj = Utrue_subj / np.sqrt(np.sum(Utrue_subj ** 2)) \
            * np.sqrt(Utrue_subj.size)
        Utrue.append(Utrue_subj)
    Utrue = np.array(Utrue)

    # run the fmri simulation
    U = []
    residuals = []
    timecourses = []
    for i_subj in range(n_subj):
        Usamps = []
        res_subj = []
        for iSamp in range(n_repeat):
            design = dnn.generate_design_random(
                len(stim_list), repeats=1, duration=duration, pause=pause,
                endzeros=endzeros)
            if use_cor_noise:
                timecourse = dnn.generate_timecourse(
                    design, Utrue[i_subj],
                    sigma_noise, resolution=resolution, ar_coeff=ar_coeff,
                    sigmaP=sigmaP[i_subj])
            else:
                timecourse = dnn.generate_timecourse(
                    design, Utrue[i_subj],
                    sigma_noise, resolution=resolution, ar_coeff=ar_coeff,
                    sigmaP=None)
            Usamp = estimate_betas(design, timecourse)
            timecourses.append(timecourse)
            Usamps.append(Usamp)
            res_subj.append(get_residuals(design, timecourse, Usamp,
                                          resolution=resolution))
        res_subj = np.concatenate(res_subj, axis=0)
        residuals.append(res_subj)
        U.append(np.array(Usamps))
    residuals = np.array(residuals)
    U = np.array(U)

    # calculate RDMs
    data = []
    desc = {'stim': np.tile(np.arange(n_stim), n_repeat),
            'repeat': np.repeat(np.arange(n_repeat), n_stim)}
    for i_subj in range(U.shape[0]):
        u_subj = U[i_subj, :, :n_stim, :].reshape(n_repeat * n_stim,
                                                  n_voxel)
        data.append(rsatoolbox.data.Dataset(u_subj, obs_descriptors=desc))
    if noise_type == 'eye':
        noise = None
    elif noise_type == 'residuals':
        noise = rsatoolbox.data.prec_from_residuals(residuals)
    rdms = rsatoolbox.rdm.calc_rdm(data, method=rdm_type, descriptor='stim',
                                   cv_descriptor='repeat', noise=noise)
    # plotting
    # true activations
    for i_sub in range(n_subj):
        plt.figure()
        ax = plt.axes()
        plt.imshow(Utrue[i_sub], cmap='Greys')
        ax.xaxis.set_ticks([])
        ax.yaxis.set_ticks([])
        plt.tight_layout()
        plt.savefig(os.path.join(folder_out, 'true_activations_%d.pdf' % i_sub))

    # noisy activations
    for i_sub in range(n_subj):
        plt.figure()
        ax = plt.axes()
        plt.imshow(U[i_sub, 0, :n_stim], cmap='Greys')
        ax.xaxis.set_ticks([])
        ax.yaxis.set_ticks([])
        plt.tight_layout()
        plt.savefig(os.path.join(folder_out, 'activations_%d.pdf' % i_sub))

    # RDMs
    matrices = rdms.get_matrices()
    for i_sub in range(n_subj):
        plt.figure()
        ax = plt.axes()
        plt.imshow(matrices[i_sub], cmap='Greys')
        ax.xaxis.set_ticks([])
        ax.yaxis.set_ticks([])
        plt.tight_layout()
        plt.savefig(os.path.join(folder_out, 'rdm_%d.pdf' % i_sub))

    # images used
    for i_stim in range(n_stim):
        stim_list[i_stim].save(
            os.path.join(folder_out, 'image%d.png' % i_stim))

    # timecourses
    for i_sub in range(n_subj):
        plt.figure()
        ax = plt.axes()
        plt.imshow(timecourses[i_sub].T, cmap='Greys')
        ax.xaxis.set_ticks([])
        ax.yaxis.set_ticks([])
        plt.tight_layout()
        plt.savefig(os.path.join(folder_out, 'timecourses_%d.pdf' % i_sub))
