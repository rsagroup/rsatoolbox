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
import rsatoolbox


def load_comp(folder):
    """this function loads all comparison results from a folder and puts
    them into a long style matrix, i.e. one p-value per row with the
    metainfo added into the other rows. The final table has the format:
        p_value | method | bootstrap-type | test-type | number of subjects |
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
        'fancy' = 3
        'fancyboot' = 4
        'fix' = 5

    test-type:
        'perc' = 0
        't' = 1
        'ranksum' = 2

    """
    table = []
    for p in pathlib.Path(folder).glob("p_*"):
        ps = np.load(p)
        split = p.name.split("_")
        if split[1] == "corr":
            method = 0
        elif split[1] == "cosine":
            method = 1
        elif split[1] == "spearman":
            method = 2
        elif split[1] == "rho_a":
            method = 3
        if split[2] == "both":
            boot = 0
        elif split[2] == "rdm":
            boot = 1
        elif split[2] == "pattern":
            boot = 2
        elif split[2] == "fancy":
            boot = 3
        elif split[2] == "fancyboot":
            boot = 4
        elif split[2] == "fix":
            boot = 5
        if split[3] == "t":
            test_type = 1
        elif split[3] == "ranksum":
            test_type = 2
        else:  # should be 'perc'
            test_type = 0
        if split[3] == "True" or split[4] == "True":
            boot_noise_ceil = True
        else:
            boot_noise_ceil = False
        if folder == "comp_noise":
            ps = ps[0]
        n_cond = int(split[-5])
        n_subj = int(split[-4])
        n_voxel = int(split[-3])
        sigma_noise = float(split[-2])
        idx = int(split[-1][:-4])
        desc = np.array(
            [
                [
                    method,
                    boot,
                    test_type,
                    n_subj,
                    n_cond,
                    n_voxel,
                    boot_noise_ceil,
                    sigma_noise,
                    idx,
                ]
            ]
        )
        desc = np.repeat(desc, len(ps), axis=0)
        new_ps = np.concatenate((np.array([ps]).T, desc), axis=1)
        table.append(new_ps)
    table = np.concatenate(table, axis=0)
    return table


def get_fname_base(
    simulation_folder,
    n_voxel,
    n_subj,
    n_repeat,
    sd,
    duration,
    pause,
    endzeros,
    use_cor_noise,
    resolution,
    sigma_noise,
    ar_coeff,
    layer=None,
    variation=None,
):
    """generates the filename base from parameters"""
    if layer is None:
        l_text = "/layer%02d"
    else:
        l_text = "/layer%02d" % layer
    if variation:
        fname_base = (
            simulation_folder
            + l_text
            + (
                "/pars_%03d_%02d_%02d_%.3f_%s/"
                % (n_voxel, n_subj, n_repeat, sd, variation)
            )
            + (
                "fmri_%02d_%02d_%03d_%s_%d_%.2f_%.2f/"
                % (
                    duration,
                    pause,
                    endzeros,
                    use_cor_noise,
                    resolution,
                    sigma_noise,
                    ar_coeff,
                )
            )
        )
    else:
        fname_base = (
            simulation_folder
            + l_text
            + ("/pars_%03d_%02d_%02d_%.3f/" % (n_voxel, n_subj, n_repeat, sd))
            + (
                "fmri_%02d_%02d_%03d_%s_%d_%.2f_%.2f/"
                % (
                    duration,
                    pause,
                    endzeros,
                    use_cor_noise,
                    resolution,
                    sigma_noise,
                    ar_coeff,
                )
            )
        )
    return fname_base


def get_resname(
    boot_type,
    rdm_type,
    model_type,
    rdm_comparison,
    noise_type,
    n_stim,
    k_pattern,
    k_rdm,
    smoothing=-1,
):
    if k_pattern is None and k_rdm is None:
        res_name = "results_%s_%s_%s_%s_%s_%d" % (
            boot_type,
            rdm_type,
            model_type,
            rdm_comparison,
            noise_type,
            n_stim,
        )
    elif k_pattern is None:
        res_name = "results_%s_%s_%s_%s_%s_%d_None_%d" % (
            boot_type,
            rdm_type,
            model_type,
            rdm_comparison,
            noise_type,
            n_stim,
            k_rdm,
        )
    elif k_rdm is None:
        res_name = "results_%s_%s_%s_%s_%s_%d_%d_None" % (
            boot_type,
            rdm_type,
            model_type,
            rdm_comparison,
            noise_type,
            n_stim,
            k_pattern,
        )
    else:
        res_name = "results_%s_%s_%s_%s_%s_%d_%d_%d" % (
            boot_type,
            rdm_type,
            model_type,
            rdm_comparison,
            noise_type,
            n_stim,
            k_pattern,
            k_rdm,
        )
    if smoothing is not None:
        res_name = res_name + "_%.2f" % (smoothing)
    return res_name


def get_stimuli_92():
    stimuli = []
    for i_stim in range(92):
        im = PIL.Image.open("96Stimuli/stimulus%d.tif" % (i_stim + 1))
        stimuli.append(im)
    return stimuli


def get_stimuli_96():
    stimuli = []
    for i_stim in range(96):
        im = PIL.Image.open("96Stimuli/stimulus%d.tif" % (i_stim + 1))
        stimuli.append(im)
    return stimuli


def get_stimuli_ecoset(ecoset_path, stim_paths):
    """loads a list of images from the ecoset folder"""
    stimuli = []
    for i_stim, stim_path in enumerate(stim_paths):
        im = PIL.Image.open(os.path.join(ecoset_path, stim_path))
        stimuli.append(im)
    return stimuli


def run_inference(
    model,
    rdms,
    method,
    bootstrap,
    boot_noise_ceil=False,
    k_rdm=None,
    k_pattern=None,
    n_cv=2,
    N=1000,
):
    """runs a run of inference

    Args:
        model(rsatoolbox.model.Model): the model(s) to be tested
        rdms(rsatoolbox.rdm.Rdms): the data
        method(String): rdm comparison method
        bootstrap(String): Bootstrapping method:
            pattern: rsatoolbox.inference.eval_bootstrap_pattern
            rdm: rsatoolbox.inference.eval_bootstrap_rdm
            both: rsatoolbox.inference.eval_bootstrap
            crossval: rsatoolbox.inference.bootstrap_crossval
            crossval_pattern: rsatoolbox.inference.bootstrap_crossval(k_rdm=1)
            crossval_rdms: rsatoolbox.inference.bootstrap_crossval(k_pattern=1)
            fancy: rsatoolbox.inference.eval_fancy
            fix: rsatoolbox.inference.eval_fix
    """
    if bootstrap == "pattern":
        results = rsatoolbox.inference.eval_bootstrap_pattern(
            model, rdms, boot_noise_ceil=boot_noise_ceil, method=method, N=N
        )
    elif bootstrap == "rdm":
        results = rsatoolbox.inference.eval_bootstrap_rdm(
            model, rdms, boot_noise_ceil=boot_noise_ceil, method=method, N=N
        )
    elif bootstrap == "both":
        results = rsatoolbox.inference.eval_bootstrap(
            model, rdms, boot_noise_ceil=boot_noise_ceil, method=method, N=N
        )
    elif bootstrap == "crossval":
        results = rsatoolbox.inference.bootstrap_crossval(
            model, rdms, method=method, k_pattern=k_pattern, k_rdm=k_rdm, n_cv=n_cv, N=N
        )
    elif bootstrap == "crossval_pattern":
        results = rsatoolbox.inference.bootstrap_crossval(
            model, rdms, method=method, k_rdm=1, k_pattern=k_pattern, n_cv=n_cv, N=N
        )
    elif bootstrap == "crossval_rdms":
        results = rsatoolbox.inference.bootstrap_crossval(
            model, rdms, method=method, k_pattern=1, k_rdm=k_rdm, n_cv=n_cv, N=N
        )
    elif bootstrap == "fancy":
        results = rsatoolbox.inference.eval_dual_bootstrap(
            model, rdms, method=method, k_pattern=k_pattern, k_rdm=k_rdm, n_cv=n_cv, N=N
        )
    elif bootstrap == "fancyboot":
        results = rsatoolbox.inference.eval_dual_bootstrap(
            model, rdms, method=method, k_pattern=1, k_rdm=1, n_cv=n_cv, N=N
        )
    elif bootstrap == "fix":
        results = rsatoolbox.inference.eval_fixed(model, rdms, method=method)
    return results


def parse_pars(pars_string):
    split = pars_string.split("_")
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
    split = fmri_string.split("_")
    duration = int(split[1])
    pause = int(split[2])
    endzeros = int(split[3])
    if split[4] == "True":
        use_cor_noise = True
    else:
        use_cor_noise = False
    resolution = float(split[5])
    sigma_noise = float(split[6])
    ar_coeff = float(split[7])
    return duration, pause, endzeros, use_cor_noise, resolution, sigma_noise, ar_coeff


def parse_results(res_string):
    split = res_string.split("_")
    boot_type = split[1]
    rdm_type = split[2]
    if split[4] in ["full", "mean", "average", "both", "avgfull"]:
        model_type = split[3] + "_" + split[4]
    else:
        model_type = split[3]
    if split[-3] == "cov":
        rdm_comparison = split[-4] + "_" + split[-3]
    else:
        rdm_comparison = split[-3]
    noise_type = split[-2]
    n_stim = int(split[-1])
    return boot_type, rdm_type, model_type, rdm_comparison, noise_type, n_stim
