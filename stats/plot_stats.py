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
import seaborn as sns
import pandas as pd
from helpers import get_fname_base
import pyrsa


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
    results = pyrsa.inference.load_results(res_path + '/res%04d.hdf5' % idx)
    pyrsa.vis.plot_model_comparison(results)


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


def plot_comp(data, alpha=0.05):
    """ plots comp check data
    """
    # methods = np.unique(data[:, 1])
    boots = np.unique(data[:, 2])
    n_subj = np.unique(data[:, 3])
    n_cond = np.unique(data[:, 4])
    n_voxel = np.unique(data[:, 5])
    # boot_noise = np.unique(data[:, 6])
    # sigmas = np.unique(data[:, 7])
    idx = np.unique(data[:, 8])
    props = np.nan * np.empty((len(boots), len(n_subj), len(n_cond),
                               len(n_voxel), len(idx)))
    for i_boot, boot in enumerate(boots):
        for i_subj, n_sub in enumerate(n_subj):
            for i_cond, cond in enumerate(n_cond):
                for i_vox, vox in enumerate(n_voxel):
                    for i, _ in enumerate(idx):
                        dat = data[data[:, 2] == boot, :]
                        dat = dat[dat[:, 3] == n_sub, :]
                        dat = dat[dat[:, 4] == cond, :]
                        dat = dat[dat[:, 5] == vox, :]
                        dat = dat[dat[:, 8] == idx[i], :]
                        if len(dat) > 0:
                            prop = (np.sum(dat[:, 0] > (1 - alpha))
                                    / len(dat))
                            props[i_boot, i_subj, i_cond, i_vox, i] = prop
                        else:
                            props[i_boot, i_subj, i_cond, i_vox, i] = np.nan
    # First plot: barplot + scatter for each type of bootstrap
    plt.figure()
    ax = plt.subplot(1, 1, 1)
    for i in range(len(boots)):
        plt.bar(i, np.mean(props[i]))
        plt.plot(np.repeat(i, props[i].size)
                 + 0.1 * np.random.randn(props[i].size),
                 props[i].flatten(), 'k.')
    plt.plot([-0.5, 2.5], [alpha, alpha], 'k--')
    plt.xticks([0, 1, 2], ['both', 'rdm', 'pattern'])
    plt.ylabel('Proportion significant')
    plt.xlabel('bootstrap method')
    # Second plot: plot against n_subj
    p_max = np.nanmax(props)
    plt.figure(figsize=(12, 5))
    for i in range(len(boots)):
        ax = plt.subplot(1, 3, i+1)
        h0 = plt.plot(np.arange(len(n_subj)), props[i, :, 0, 0, :], '.',
                      color=[0.5, 0, 0])
        h1 = plt.plot(np.arange(len(n_subj)), props[i, :, 1, 0, :], '.',
                      color=[0.5, 0.2, 0.3])
        h2 = plt.plot(np.arange(len(n_subj)), props[i, :, 2, 0, :], '.',
                      color=[0.5, 0.4, 0.7])
        if len(n_cond) > 3:
            h3 = plt.plot(np.arange(len(n_subj)), props[i, :, 3, 0, :], '.',
                          color=[0.5, 0.6, 1])
        if i == 0:
            plt.title('both', fontsize=18)
            plt.ylabel('Proportion significant', fontsize=18)
        elif i == 1:
            plt.title('rdm', fontsize=18)
        elif i == 2:
            plt.title('pattern', fontsize=18)
            if len(n_cond) > 3:
                plt.legend([h0[0], h1[0], h2[0], h3[0]], n_cond.astype('int'),
                           frameon=False, title='# of patterns')
            else:
                plt.legend([h0[0], h1[0], h2[0]], n_cond.astype('int'),
                           frameon=False, title='# of patterns')
        plt.xticks(np.arange(len(n_subj)), n_subj.astype('int'))
        plt.yticks([0, alpha, 2*alpha, 3*alpha])
        plt.ylim([0, p_max + 0.01])
        plt.xlim([-1, len(n_subj)])
        plt.plot([-1, len(n_subj)], [alpha, alpha], 'k--')
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        plt.xlabel('# of rdms', fontsize=18)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    # Third plot: plot against n_pattern
    plt.figure(figsize=(12, 5))
    for i in range(len(boots)):
        ax = plt.subplot(1, 3, i+1)
        h0 = plt.plot(np.arange(len(n_cond)), props[i, 0, :, 0, :], '.',
                      color=[0.5, 0, 0])
        h1 = plt.plot(np.arange(len(n_cond)), props[i, 1, :, 0, :], '.',
                      color=[0.5, 0.2, 0.3])
        h2 = plt.plot(np.arange(len(n_cond)), props[i, 2, :, 0, :], '.',
                      color=[0.5, 0.4, 0.7])
        h3 = plt.plot(np.arange(len(n_cond)), props[i, 3, :, 0, :], '.',
                      color=[0.5, 0.6, 1])
        if i == 0:
            plt.title('both', fontsize=18)
            plt.ylabel('Proportion significant', fontsize=18)
        elif i == 1:
            plt.title('rdm', fontsize=18)
        elif i == 2:
            plt.title('pattern', fontsize=18)
            plt.legend([h0[0], h1[0], h2[0], h3[0]], n_subj.astype('int'),
                       frameon=False, title='# of rdms')
        plt.xticks(np.arange(len(n_cond)), n_cond.astype('int'))
        plt.yticks([0, alpha, 2*alpha, 3*alpha])
        plt.ylim([0, p_max + 0.01])
        plt.xlim([-1, len(n_cond)])
        plt.plot([-1, len(n_cond)], [alpha, alpha], 'k--')
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        plt.xlabel('# of patterns', fontsize=18)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)


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
    std_var = np.nanvar(stds, axis=1)
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
                         order=[5, 20, 80])
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
