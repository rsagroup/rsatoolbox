#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 15 08:01:30 2020

@author: heiko
"""

import os
import PIL
import pathlib
import numpy as np
import pyrsa


def load_comp(folder):
    """ this function loads all comparison results from a folder and puts
    them into a long style matrix, i.e. one p-value per row with the
    metainfo added into the other rows. The final table has the format:
        p_value | method | bootstrap-type | number of subjects |
        number of patterns | number of voxels | boot_noise_ceil|
        sigma_noise | idx
    methods:
        'corr' = 0
        'cosine' = 1
        'spearman' = 2
        'rho_a' = 3
        ''
    bootstrap-type:
        'both' = 0
        'rdm' = 1
        'pattern' = 2

    """
    table = []
    for p in pathlib.Path(folder).glob('p_*'):
        ps = np.load(p)
        split = p.name.split('_')
        if split[1] == 'corr':
            method = 0
        elif split[1] == 'cosine':
            method = 1
        elif split[1] == 'spearman':
            method = 2
        elif split[1] == 'rho_a':
            method = 3
        if split[2] == 'both':
            boot = 0
        elif split[2] == 'rdm':
            boot = 1
        elif split[2] == 'pattern':
            boot = 2
        elif split[2] == 'fancy':
            boot = 3
        elif split[2] == 'fancyboot':
            boot = 4
        elif split[2] == 'fix':
            boot = 5
        if split[3] == 't':
            test_type = 1
        elif split[3] == 'fix':
            test_type = 2
        else:
            test_type = 0
        if split[3] == 'True' or split[4] == 'True':
            boot_noise_ceil = True
        else:
            boot_noise_ceil = False
        if folder == 'comp_noise':
            ps = ps[0]
        n_cond = int(split[-5])
        n_subj = int(split[-4])
        n_voxel = int(split[-3])
        sigma_noise = float(split[-2])
        idx = int(split[-1][:-4])
        desc = np.array([[method, boot, test_type, n_subj, n_cond, n_voxel,
                          boot_noise_ceil, sigma_noise, idx]])
        desc = np.repeat(desc, len(ps), axis=0)
        new_ps = np.concatenate((np.array([ps]).T, desc), axis=1)
        table.append(new_ps)
    table = np.concatenate(table, axis=0)
    return table


def get_fname_base(simulation_folder, n_voxel, n_subj, n_repeat, sd,
                   duration, pause, endzeros, use_cor_noise, resolution,
                   sigma_noise, ar_coeff, layer=None, variation=None):
    """ generates the filename base from parameters """
    if layer is None:
        l_text = '/layer%02d'
    else:
        l_text = '/layer%02d' % layer
    if variation:
        fname_base = simulation_folder + l_text \
            + ('/pars_%03d_%02d_%02d_%.3f_%s/' % (
                n_voxel, n_subj, n_repeat, sd, variation)) \
            + ('fmri_%02d_%02d_%03d_%s_%d_%.2f_%.2f/' % (
                duration, pause, endzeros, use_cor_noise, resolution,
                sigma_noise, ar_coeff))
    else:
        fname_base = simulation_folder + l_text \
            + ('/pars_%03d_%02d_%02d_%.3f/' % (
                n_voxel, n_subj, n_repeat, sd)) \
            + ('fmri_%02d_%02d_%03d_%s_%d_%.2f_%.2f/' % (
                duration, pause, endzeros, use_cor_noise, resolution,
                sigma_noise, ar_coeff))
    return fname_base


def get_resname(boot_type, rdm_type, model_type, rdm_comparison, noise_type,
                n_stim, k_pattern, k_rdm):
    if k_pattern is None and k_rdm is None:
        res_name = 'results_%s_%s_%s_%s_%s_%d' % (
            boot_type, rdm_type, model_type, rdm_comparison, noise_type,
            n_stim)
    elif k_pattern is None:
        res_name = 'results_%s_%s_%s_%s_%s_%d_None_%d' % (
            boot_type, rdm_type, model_type, rdm_comparison, noise_type,
            n_stim, k_rdm)
    elif k_rdm is None:
        res_name = 'results_%s_%s_%s_%s_%s_%d_%d_None' % (
            boot_type, rdm_type, model_type, rdm_comparison, noise_type,
            n_stim, k_pattern)
    else:
        res_name = 'results_%s_%s_%s_%s_%s_%d_%d_%d' % (
            boot_type, rdm_type, model_type, rdm_comparison, noise_type,
            n_stim, k_pattern, k_rdm)
    return res_name


def get_stimuli_92():
    stimuli = []
    for i_stim in range(92):
        im = PIL.Image.open('96Stimuli/stimulus%d.tif' % (i_stim+1))
        stimuli.append(im)
    return stimuli


def get_stimuli_96():
    stimuli = []
    for i_stim in range(96):
        im = PIL.Image.open('96Stimuli/stimulus%d.tif' % (i_stim+1))
        stimuli.append(im)
    return stimuli


def get_stimuli_ecoset(ecoset_path, stim_paths):
    """ loads a list of images from the ecoset folder """
    stimuli = []
    for i_stim, stim_path in enumerate(stim_paths):
        im = PIL.Image.open(os.path.join(ecoset_path, stim_path))
        stimuli.append(im)
    return stimuli


def run_inference(model, rdms, method, bootstrap, boot_noise_ceil=False,
                  k_rdm=None, k_pattern=None):
    """ runs a run of inference

    Args:
        model(pyrsa.model.Model): the model(s) to be tested
        rdms(pyrsa.rdm.Rdms): the data
        method(String): rdm comparison method
        bootstrap(String): Bootstrapping method:
            pattern: pyrsa.inference.eval_bootstrap_pattern
            rdm: pyrsa.inference.eval_bootstrap_rdm
            both: pyrsa.inference.eval_bootstrap
            crossval: pyrsa.inference.bootstrap_crossval
            crossval_pattern: pyrsa.inference.bootstrap_crossval(k_rdm=1)
            crossval_rdms: pyrsa.inference.bootstrap_crossval(k_pattern=1)
            fancy: pyrsa.inference.eval_fancy
            fix: pyrsa.inference.eval_fix
    """
    if bootstrap == 'pattern':
        results = pyrsa.inference.eval_bootstrap_pattern(
            model, rdms,
            boot_noise_ceil=boot_noise_ceil, method=method)
    elif bootstrap == 'rdm':
        results = pyrsa.inference.eval_bootstrap_rdm(
            model, rdms,
            boot_noise_ceil=boot_noise_ceil, method=method)
    elif bootstrap == 'both':
        results = pyrsa.inference.eval_bootstrap(
            model, rdms,
            boot_noise_ceil=boot_noise_ceil, method=method)
    elif bootstrap == 'crossval':
        results = pyrsa.inference.bootstrap_crossval(
            model, rdms, method=method, k_pattern=k_pattern, k_rdm=k_rdm)
    elif bootstrap == 'crossval_pattern':
        results = pyrsa.inference.bootstrap_crossval(
            model, rdms, method=method, k_rdm=1, k_pattern=k_pattern)
    elif bootstrap == 'crossval_rdms':
        results = pyrsa.inference.bootstrap_crossval(
            model, rdms, method=method, k_pattern=1, k_rdm=k_rdm)
    elif bootstrap == 'fancy':
        results = pyrsa.inference.eval_fancy(
            model, rdms, method=method, k_pattern=k_pattern, k_rdm=k_rdm)
    elif bootstrap == 'fancyboot':
        results = pyrsa.inference.eval_fancy(
            model, rdms, method=method, k_pattern=1, k_rdm=1)
    elif bootstrap == 'fix':
        results = pyrsa.inference.eval_fixed(
            model, rdms, method=method)
    return results


def parse_pars(pars_string):
    split = pars_string.split('_')
    n_voxel = int(split[1])
    n_subj = int(split[2])
    n_rep = int(split[3])
    sd = float(split[4])
    if len(split) > 5:
        variation = split[5]
    else:
        variation = None
    return n_voxel, n_subj, n_rep, sd, variation


def parse_fmri(fmri_string):
    split = fmri_string.split('_')
    duration = int(split[1])
    pause = int(split[2])
    endzeros = int(split[3])
    if split[4] == 'True':
        use_cor_noise = True
    else:
        use_cor_noise = False
    resolution = float(split[5])
    sigma_noise = float(split[6])
    ar_coeff = float(split[7])
    return duration, pause, endzeros, use_cor_noise, resolution, \
        sigma_noise, ar_coeff


def parse_results(res_string):
    split = res_string.split('_')
    boot_type = split[1]
    rdm_type = split[2]
    if len(split) == 8:
        model_type = split[3] + '_' + split[4]
    else:
        model_type = split[3]
    rdm_comparison = split[-3]
    noise_type = split[-2]
    n_stim = int(split[-1])
    return boot_type, rdm_type, model_type, rdm_comparison, noise_type, n_stim