#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 22:29:12 2019

@author: heiko
Functions to check the statistical integrity of the toolbox
"""

import os
import numpy as np
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt
import tqdm
import scipy.signal as signal
from hrf import spm_hrf
from scipy.ndimage import gaussian_filter as gaussian_filter
import pyrsa
import nn_simulations as dnn
import pathlib


def get_stimuli_92():
    import PIL
    stimuli = []
    for i_stim in range(92):
        im = PIL.Image.open('96Stimuli/stimulus%d.tif' % (i_stim+1))
        stimuli.append(im)
    return stimuli


def get_stimuli_96():
    import PIL
    stimuli = []
    for i_stim in range(96):
        im = PIL.Image.open('96Stimuli/stimulus%d.tif' % (i_stim+1))
        stimuli.append(im)
    return stimuli


def estimate_betas(design, timecourse, hrf=None, resolution=2):
    if hrf is None:
        t = np.arange(0, 30, resolution)
        hrf = spm_hrf(t)
    hrf = np.array([hrf]).transpose()
    design = signal.convolve(design, hrf, mode='full')[:design.shape[0]]
    solver = np.linalg.inv(np.matmul(design.transpose(), design))
    beta = np.matmul(np.matmul(solver, design.transpose()), timecourse)
    return beta


def get_residuals(design, timecourse, beta, resolution=2, hrf=None):
    if hrf is None:
        t = np.arange(0, 30, resolution)
        hrf = spm_hrf(t)
    hrf = np.array([hrf]).transpose()
    design = signal.convolve(design, hrf, mode='full')[:design.shape[0]]
    residuals = timecourse - np.matmul(design, beta)
    return residuals


def get_residuals_cross(designs, timecourses, betas, resolution=2, hrf=None):
    residuals = np.zeros_like(timecourses)
    for iCross in range(len(designs)):
        selected = np.ones(len(designs), 'bool')
        selected[iCross] = 0
        beta = np.mean(betas[selected], axis=0)
        residuals[iCross] = get_residuals(designs[iCross],
                                          timecourses[iCross],
                                          beta,
                                          resolution=resolution,
                                          hrf=hrf)
    return residuals


def run_inference(model, rdms, method, bootstrap, boot_noise_ceil=False):
    """ runs a run of inference
    
    Args:
        model(pyrsa.model.Model): the model(s) to be tested
        rdms(pyrsa.rdm.Rdms): the data
        method(String): rdm comparison method
        bootstrap(String): Bootstrapping method:
            pattern: pyrsa.inference.eval_bootstrap_pattern
            rdm: pyrsa.inference.eval_bootstrap_rdm
            boot: pyrsa.inference.eval_bootstrap
            crossval: pyrsa.inference.bootstrap_crossval
            crossval_pattern: pyrsa.inference.bootstrap_crossval(k_rdm=1)
            rdmcrossval_rdms pyrsa.inference.bootstrap_crossval(k_pattern=1)
    """
    if bootstrap == 'pattern':
        results = pyrsa.inference.eval_bootstrap_pattern(model, rdms,
            boot_noise_ceil=boot_noise_ceil, method=method)
    elif bootstrap == 'rdm':
        results = pyrsa.inference.eval_bootstrap_rdm(model, rdms,
            boot_noise_ceil=boot_noise_ceil, method=method)
    elif bootstrap == 'boot':
        results = pyrsa.inference.eval_bootstrap(model, rdms,
            boot_noise_ceil=boot_noise_ceil, method=method)
    elif bootstrap == 'crossval':
        results = pyrsa.inference.bootstrap_crossval(model, rdms,
            boot_noise_ceil=boot_noise_ceil, method=method)
    elif bootstrap == 'crossval_pattern':
        results = pyrsa.inference.bootstrap_crossval(model, rdms,
            boot_noise_ceil=boot_noise_ceil, method=method, k_rdm=1)
    elif bootstrap == 'crossval_rdms':
        results = pyrsa.inference.bootstrap_crossval(model, rdms,
            boot_noise_ceil=boot_noise_ceil, method=method, k_pattern=1)
    return results


def check_compare_to_zero(model, n_voxel=100, n_subj=10, n_sim=1000,
                          method='corr', bootstrap='pattern'):
    """ runs simulations for comparison to zero
    It compares whatever model you pass to pure noise data, generated
    as independent normal noise for the voxels and subjects.
    
    Args:
        model(pyrsa.model.Model): the model to be tested against
        n_voxel(int): number of voxels to be simulated per subject
        n_subj(int): number of subjects to be simulated
        n_sim(int): number of simulations to be performed

    """
    n_cond = int(model.n_cond)
    p = np.empty(n_sim)
    for i_sim in range(n_sim):
        raw_u = np.random.randn(n_subj, n_cond, n_voxel)
        data = []
        for i_subj in range(n_subj):
            dat = pyrsa.data.Dataset(raw_u[i_subj])
            data.append(dat)
        rdms = pyrsa.rdm.calc_rdm(data)
        results = run_inference(model, rdms, method, bootstrap)
        idx_valid = ~np.isnan(results.evaluations)
        p[i_sim] = np.sum(results.evaluations[idx_valid] > 0) \
            / np.sum(idx_valid)
    return p


def save_compare_to_zero(idx, n_voxel=100, n_subj=10, n_cond=5,
                         method='corr', bootstrap='pattern',
                         folder='comp_zero'):
    """ saves the results of a simulation to a file 
    """
    if not os.path.isdir(folder):
        os.mkdir(folder)
    fname = folder + os.path.sep + 'p_%s_%s_%d_%d_%d_%03d.npy' % (method,
        bootstrap, n_cond, n_subj, n_voxel, idx)
    model_u = np.random.randn(n_cond, n_voxel)
    model_dat = pyrsa.data.Dataset(model_u)
    model_rdm = pyrsa.rdm.calc_rdm(model_dat)
    model = pyrsa.model.ModelFixed('test', model_rdm)
    p = check_compare_to_zero(model, n_voxel=n_voxel, n_subj=n_subj,
                              method=method, bootstrap=bootstrap)
    np.save(fname, p)


def check_compare_models(model1, model2, n_voxel=100, n_subj=10, n_sim=1000,
                         method='corr', bootstrap='pattern', sigma_noise=1):
    """ runs simulations for comparison to zero
    It compares whatever model you pass to pure noise data, generated
    as independent normal noise for the voxels and subjects.
    
    Args:
        model(pyrsa.model.Model): the model to be tested against each other
        n_voxel(int): number of voxels to be simulated per subject
        n_subj(int): number of subjects to be simulated
        n_sim(int): number of simulations to be performed

    """
    assert model1.n_cond == model2.n_cond
    n_cond = int(model1.n_cond)
    rdm1 = model1.predict()
    rdm2 = model2.predict()
    rdm1 = rdm1 - np.mean(rdm1)
    rdm2 = rdm2 - np.mean(rdm2)
    rdm1 = rdm1 / np.std(rdm1)
    rdm2 = rdm2 / np.std(rdm2)
    target_rdm = (rdm1 + rdm2) / 2
    target_rdm = target_rdm - np.min(target_rdm)
    # the following guarantees the triangle inequality
    # without this the generation fails
    target_rdm = target_rdm + np.max(target_rdm)
    t_rdm = pyrsa.rdm.RDMs(target_rdm)
    D = squareform(target_rdm)
    H = pyrsa.util.matrix.centering(D.shape[0])
    G = -0.5 * (H @ D @ H)
    U0 = pyrsa.simulation.make_signal(G, n_voxel, make_exact=True)
    dat0 = pyrsa.data.Dataset(U0)
    rdm0 = pyrsa.rdm.calc_rdm(dat0)
    p = np.empty(n_sim)
    for i_sim in range(n_sim):
        raw_u = U0 + sigma_noise * np.random.randn(n_subj, n_cond, n_voxel)
        data = []
        for i_subj in range(n_subj):
            dat = pyrsa.data.Dataset(raw_u[i_subj])
            data.append(dat)
        rdms = pyrsa.rdm.calc_rdm(data)
        results = run_inference([model1, model2], rdms, method, bootstrap)
        idx_valid = ~np.isnan(results.evaluations[:, 0])
        p[i_sim] = np.sum(results.evaluations[idx_valid, 0] > 
                          results.evaluations[idx_valid, 1]) / np.sum(idx_valid)
    return p


def save_compare_models(idx, n_voxel=100, n_subj=10, n_cond=5,
                        method='corr', bootstrap='pattern',
                        folder='comp_model'):
    """ saves the results of a simulation to a file 
    """
    if not os.path.isdir(folder):
        os.mkdir(folder)
    fname = folder + os.path.sep + 'p_%s_%s_%d_%d_%d_%03d.npy' % (method,
        bootstrap, n_cond, n_subj, n_voxel, idx)
    model1_u = np.random.randn(n_cond, n_voxel)
    model1_dat = pyrsa.data.Dataset(model1_u)
    model1_rdm = pyrsa.rdm.calc_rdm(model1_dat)
    model1 = pyrsa.model.ModelFixed('test1', model1_rdm)
    model2_u = np.random.randn(n_cond, n_voxel)
    model2_dat = pyrsa.data.Dataset(model2_u)
    model2_rdm = pyrsa.rdm.calc_rdm(model2_dat)
    model2 = pyrsa.model.ModelFixed('test2', model2_rdm)
    p = check_compare_models(model1, model2, n_voxel=n_voxel, n_subj=n_subj,
                              method=method, bootstrap=bootstrap)
    np.save(fname, p)


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
    ax = plt.subplot(1,1,1)
    plt.plot(bins, n_binned.T, )
    plt.plot([0,1],[0,1], 'k--')
    ax.set_aspect('equal', 'box')
    plt.xlabel('alpha')
    plt.ylabel('proportion p<alpha')
    plt.xlim([0,1])
    plt.ylim([0,1])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.figure()
    plt.plot(n_significant, 'k.')
    plt.plot([0 - 0.5, len(n_significant) - 0.5], [50, 50], 'k--')
    plt.ylim(bottom=0)


def save_simulated_data_dnn(model=dnn.get_default_model(), layer=2, sd=0.05,
                            stimList=get_stimuli_96(), n_voxel=100, n_subj=10,
                            simulation_folder='sim', n_sim=1000, n_repeat=2,
                            duration=1, pause=1, endzeros=25,
                            use_cor_noise=True, resolution=2,
                            sigma_noise = 2, ar_coeff = .5):
    fname_base = get_fname_base(simulation_folder=simulation_folder,
                                layer=layer, n_voxel=n_voxel, n_subj=n_subj,
                                n_repeat=n_repeat, sd=sd, duration=duration,
                                pause=pause, endzeros=endzeros,
                                use_cor_noise=use_cor_noise,
                                resolution=resolution,
                                sigma_noise=sigma_noise,
                                ar_coeff=ar_coeff)
    if not os.path.isdir(fname_base):
        os.makedirs(fname_base)
    for i in tqdm.trange(n_sim):
        Utrue = []
        sigmaP = []
        indices_space = []
        weights = []
        U = []
        des = []
        tim = []
        residuals = []
        for i_subj in range(n_subj):
            (Utrue_subj,sigmaP_subj, indices_space_subj, weights_subj) = \
                dnn.get_sampled_representations(model, layer, [sd, sd],
                                                stimList, n_voxel)
            Utrue_subj = Utrue_subj / np.sqrt(np.sum(Utrue_subj ** 2)) \
                * np.sqrt(Utrue_subj.size)
            designs = []
            timecourses = []
            Usamps = []
            res_subj = []
            for iSamp in range(n_repeat):
                design = dnn.generate_design_random(len(stimList),
                    repeats=1, duration=duration, pause=pause,
                    endzeros=endzeros)
                if use_cor_noise:
                    timecourse = dnn.generate_timecourse(design, Utrue_subj,
                        sigma_noise, resolution=resolution, ar_coeff=ar_coeff,
                        sigmaP=sigmaP_subj)
                else:
                    timecourse = dnn.generate_timecourse(design, Utrue_subj,
                        sigma_noise, resolution=resolution, ar_coeff=ar_coeff,
                        sigmaP=None)
                Usamp = estimate_betas(design, timecourse)
                designs.append(design)
                timecourses.append(timecourse)
                Usamps.append(Usamp)
                res_subj.append(get_residuals(design, timecourse, Usamp,
                                              resolution=resolution))
            res_subj = np.concatenate(res_subj, axis=0)
            residuals.append(res_subj)
            U.append(np.array(Usamps))
            des.append(np.array(designs))
            tim.append(np.array(timecourses))
            Utrue.append(Utrue_subj)
            sigmaP.append(sigmaP_subj)
            indices_space.append(indices_space_subj)
            weights.append(weights_subj)
        Utrue = np.array(Utrue)
        sigmaP = np.array(sigmaP)
        residuals = np.array(residuals)
        indices_space = np.array(indices_space)
        weights = np.array(weights)
        U = np.array(U)
        des = np.array(des)
        tim = np.array(tim)
        np.save(fname_base + 'Utrue%04d' % i, Utrue)
        np.save(fname_base + 'sigmaP%04d' % i, sigmaP)
        np.save(fname_base + 'residuals%04d' % i, residuals)
        np.save(fname_base + 'indices_space%04d' % i, indices_space)
        np.save(fname_base + 'weights%04d' % i, weights)
        np.save(fname_base + 'U%04d' % i, U)


def analyse_saved_dnn(layer=2, sd=0.05, n_voxel=100,
                      n_subj=10, simulation_folder='sim', n_sim=100, n_repeat=2,
                      duration=1, pause=1, endzeros=25, use_cor_noise=True,
                      resolution=2, sigma_noise=2, ar_coeff=0.5,
                      model_type='fixed_averagetrue',
                      rdm_comparison='cosine', n_Layer=12, k_pattern=3,
                      k_rdm=3, rdm_type='crossnobis', n_stimuli=92,
                      noise_type = 'eye'):
    fname_base = get_fname_base(simulation_folder=simulation_folder,
                                layer=layer, n_voxel=n_voxel, n_subj=n_subj,
                                n_repeat=n_repeat, sd=sd, duration=duration,
                                pause=pause, endzeros=endzeros,
                                use_cor_noise=use_cor_noise,
                                resolution=resolution,
                                sigma_noise=sigma_noise,
                                ar_coeff=ar_coeff)
    print(fname_base)
    assert os.path.isdir(fname_base), 'simulated data not found!'
    res_path = fname_base + 'results_%s_%s_%s_%s_%d_%d_%d' % (
        rdm_type, model_type, rdm_comparison, noise_type, n_stimuli,
        k_pattern, k_rdm)
    if not os.path.isdir(res_path):
        os.mkdir(res_path)
    models = []
    pat_desc = {'stim':np.arange(n_stimuli)}
    print('\n generating models\n')
    stimuli = dnn.get_stimuli_96()[:n_stimuli]
    for i_layer in tqdm.trange(n_Layer):
        if model_type == 'fixed_averagetrue':
            fname_base_l = get_fname_base(simulation_folder=simulation_folder,
                                layer=i_layer, n_voxel=n_voxel, n_subj=n_subj,
                                n_repeat=n_repeat, sd=sd, duration=duration,
                                pause=pause, endzeros=endzeros,
                                use_cor_noise=use_cor_noise,
                                resolution=resolution,
                                sigma_noise=sigma_noise,
                                ar_coeff=ar_coeff)
            rdm_true_average = 0
            for i in range(n_sim):
                Utrue = np.load(fname_base_l + 'Utrue%04d.npy' % i)
                dat_true = [pyrsa.data.Dataset(Utrue[i, :n_stimuli,:])
                            for i in range(Utrue.shape[0])]
                rdm_true = pyrsa.rdm.calc_rdm(dat_true, method='euclidean')
                rdm_mat = rdm_true.get_vectors()
                rdm_mat = rdm_mat / np.sqrt(np.mean(rdm_mat ** 2))
                rdm_true_average = rdm_true_average + np.mean(rdm_mat, 0)
            rdm = rdm_true_average / n_sim
            rdm = pyrsa.rdm.RDMs(rdm, pattern_descriptors=pat_desc)
            model = pyrsa.model.ModelFixed('Layer%02d' % i_layer, rdm)
        elif model_type == 'fixed_full':
            rdm = dnn.get_true_RDM(
                model=dnn.get_default_model(),
                layer=i_layer,
                stimuli=stimuli)
            rdm.pattern_descriptors = pat_desc
            model = pyrsa.model.ModelFixed('Layer%02d' % i_layer, rdm)
        elif model_type == 'select_full':
            smoothings = np.array([0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, np.inf])
            rdms = []
            for i_smooth in  range(len(smoothings)):
                rdm = dnn.get_true_RDM(
                    model=dnn.get_default_model(),
                    layer=i_layer,
                    stimuli=stimuli,
                    smoothing=smoothings[i_smooth],
                    average=False)
                rdm.pattern_descriptors = pat_desc
                rdms.append(rdm)
            rdms = pyrsa.rdm.concat(rdms)
            model = pyrsa.model.ModelSelect('Layer%02d' % i_layer, rdms)
        elif model_type == 'select_avg':
            smoothings = np.array([0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, np.inf])
            rdms = []
            for i_smooth in  range(len(smoothings)):
                rdm = dnn.get_true_RDM(
                    model=dnn.get_default_model(),
                    layer=i_layer,
                    stimuli=stimuli,
                    smoothing=smoothings[i_smooth],
                    average=True)
                rdm.pattern_descriptors = pat_desc
                rdms.append(rdm)
            rdms = pyrsa.rdm.concat(rdms)
            model = pyrsa.model.ModelSelect('Layer%02d' % i_layer, rdms)
        elif model_type == 'select_both':
            smoothings = np.array([0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, np.inf])
            rdms = []
            for i_smooth in  range(len(smoothings)):
                rdm = dnn.get_true_RDM(
                    model=dnn.get_default_model(),
                    layer=i_layer,
                    stimuli=stimuli,
                    smoothing=smoothings[i_smooth],
                    average=False)
                rdm.pattern_descriptors = pat_desc
                rdms.append(rdm)
                rdm = dnn.get_true_RDM(
                    model=dnn.get_default_model(),
                    layer=i_layer,
                    stimuli=stimuli,
                    smoothing=smoothings[i_smooth],
                    average=True)
                rdm.pattern_descriptors = pat_desc
                rdms.append(rdm)
            rdms = pyrsa.rdm.concat(rdms)
            model = pyrsa.model.ModelSelect('Layer%02d' % i_layer, rdms)
        elif model_type == 'interpolate_full':
            smoothings = np.array([0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, np.inf])
            rdms = []
            for i_smooth in  range(len(smoothings)):
                rdm = dnn.get_true_RDM(
                    model=dnn.get_default_model(),
                    layer=i_layer,
                    stimuli=stimuli,
                    smoothing=smoothings[i_smooth],
                    average=False)
                rdm.pattern_descriptors = pat_desc
                rdms.append(rdm)
            rdms = pyrsa.rdm.concat(rdms)
            model = pyrsa.model.ModelInterpolate('Layer%02d' % i_layer, rdms)
        elif model_type == 'interpolate_avg':
            smoothings = np.array([0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, np.inf])
            rdms = []
            for i_smooth in  range(len(smoothings)):
                rdm = dnn.get_true_RDM(
                    model=dnn.get_default_model(),
                    layer=i_layer,
                    stimuli=stimuli,
                    smoothing=smoothings[i_smooth],
                    average=True)
                rdm.pattern_descriptors = pat_desc
                rdms.append(rdm)
            rdms = pyrsa.rdm.concat(rdms)
            model = pyrsa.model.ModelInterpolate('Layer%02d' % i_layer, rdms)
        elif model_type == 'interpolate_both':
            smoothings = np.array([0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, np.inf])
            rdms = []
            for i_smooth in  range(len(smoothings)):
                rdm = dnn.get_true_RDM(
                    model=dnn.get_default_model(),
                    layer=i_layer,
                    stimuli=stimuli,
                    smoothing=smoothings[i_smooth],
                    average=True)
                rdm.pattern_descriptors = pat_desc
                rdms.append(rdm)
            for i_smooth in  range(len(smoothings)-1,-1,-1):
                rdm = dnn.get_true_RDM(
                    model=dnn.get_default_model(),
                    layer=i_layer,
                    stimuli=stimuli,
                    smoothing=smoothings[i_smooth],
                    average=False)
                rdm.pattern_descriptors = pat_desc
                rdms.append(rdm)
            rdms = pyrsa.rdm.concat(rdms)
            model = pyrsa.model.ModelInterpolate('Layer%02d' % i_layer, rdms)
        elif model_type == 'weighted_avgfull':
            rdms = []
            rdm = dnn.get_true_RDM(
                model=dnn.get_default_model(),
                layer=i_layer,
                stimuli=stimuli,
                smoothing=None,
                average=True)
            rdms.append(rdm)
            rdm = dnn.get_true_RDM(
                model=dnn.get_default_model(),
                layer=i_layer,
                stimuli=stimuli,
                smoothing=False,
                average=False)
            rdms.append(rdm)
            rdm = dnn.get_true_RDM(
                model=dnn.get_default_model(),
                layer=i_layer,
                stimuli=stimuli,
                smoothing=np.inf,
                average=False)
            rdms.append(rdm)
            rdm = dnn.get_true_RDM(
                model=dnn.get_default_model(),
                layer=i_layer,
                stimuli=stimuli,
                smoothing=np.inf,
                average=False)
            rdms.append(rdm)
            model = pyrsa.model.ModelWeighted('Layer%02d' % i_layer, rdms)
        models.append(model)
    for i in tqdm.trange(n_sim, position=1):
        U = np.load(fname_base + 'U%04d.npy' % i)
        data = []
        desc = {'stim': np.tile(np.arange(n_stimuli), n_repeat),
                'repeat': np.repeat(np.arange(n_repeat), n_stimuli)}
        for i_subj in range(U.shape[0]):
            u_subj = U[i_subj, :, :n_stimuli, :].reshape(n_repeat * n_stimuli,
                                                         n_voxel)
            data.append(pyrsa.data.Dataset(u_subj, obs_descriptors=desc))
        if noise_type == 'eye':
            noise = None
        elif noise_type == 'residuals':
            residuals = np.load(fname_base + 'residuals%04d.npy' % i)
            noise = pyrsa.data.get_prec_from_residuals(residuals)
        rdms = pyrsa.rdm.calc_rdm(data, method=rdm_type, descriptor='stim',
                                  cv_descriptor='repeat', noise=noise)
        results = pyrsa.inference.bootstrap_crossval(models, rdms,
            pattern_descriptor='stim', rdm_descriptor='index',
            k_pattern=k_pattern, k_rdm=k_rdm, method=rdm_comparison)
        results.save(res_path + '/res%04d.hdf5' % (i))


def plot_saved_dnn(layer=2, sd=0.05, n_voxel=100, idx=0,
                   n_subj=10, simulation_folder='sim', n_repeat=2,
                   duration=1, pause=1, endzeros=25, use_cor_noise=True,
                   resolution=2, sigma_noise=2, ar_coeff=0.5,
                   model_type='fixed_averagetrue',
                   rdm_comparison='cosine', n_Layer=12, k_pattern=3, k_rdm=3,
                   rdm_type='crossnobis', n_stimuli=92, fname_base=None,
                   noise_type='eye'):
    if fname_base is None:
        fname_base = get_fname_base(simulation_folder=simulation_folder,
                                    layer=layer, n_voxel=n_voxel, n_subj=n_subj,
                                    n_repeat=n_repeat, sd=sd, duration=duration,
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


def get_fname_base(simulation_folder, layer, n_voxel, n_subj, n_repeat, sd,
                   duration, pause, endzeros, use_cor_noise, resolution,
                   sigma_noise, ar_coeff):
    """ generates the filename base from parameters """
    fname_base = simulation_folder + ('/layer%02d' % layer) \
        + ('/pars_%03d_%02d_%02d_%.3f/' % (n_voxel, n_subj, n_repeat, sd)) \
        + ('fmri_%02d_%02d_%03d_%s_%d_%.2f_%.2f/' % (
            duration, pause, endzeros, use_cor_noise, resolution,
            sigma_noise, ar_coeff))
    return fname_base


def plot_saved_dnn_average(layer=2, sd=3, stimList=get_stimuli_96(),
                           n_voxel=100, n_subj=10, simulation_folder='test',
                           n_sim=100, n_repeat=2, duration=5, pause=1,
                           endzeros=25, use_cor_noise=True, resolution = 2,
                           sigma_noise=2, ar_coeff=.5, modelType = 'fixed',
                           model_rdm = 'averagetrue', n_stimuli=92,
                           rdm_comparison = 'cosine', n_Layer = 12, n_fold=5,
                           rdm_type='crossnobis', fname_base=None):
    if fname_base is None:
        fname_base = get_fname_base(simulation_folder=simulation_folder,
                                    layer=layer, n_voxel=n_voxel, n_subj=n_subj,
                                    n_repeat=n_repeat, sd=sd, duration=duration,
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
    for iSim in range(n_sim):
        ax.fill_between(np.array([0.5, n_Layer + 0.5]), noise_ceilings[iSim, 0],
                        noise_ceilings[iSim, 1], facecolor='blue',
                        alpha=1 / n_sim)
    #ax.plot(np.array([0.5,NLayer+0.5]),np.repeat(noise_ceilings[:,0],2).reshape([Nsim,2]).T,'k',alpha=.1)
    #ax.plot(np.array([0.5,NLayer+0.5]),np.repeat(noise_ceilings[:,1],2).reshape([Nsim,2]).T,'k',alpha=.1)
    ax.plot(np.arange(n_Layer) + 1 - n_fold / 20, np.mean(scores[:, :n_Layer, :],
                                                        axis=2).T,
            'k.')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel('Layer', fontsize=18)
    ax.set_title('Layer %d' % layer, fontsize=28)
    if rdm_comparison=='cosine':
        plt.ylim([0,1])
        ax.set_ylabel('Cosine Distance', fontsize=18)
    elif rdm_comparison=='eudlid':
        ax.set_ylabel('Euclidean Distance', fontsize=18)
    elif rdm_comparison=='kendall-tau':
        ax.set_ylabel('Kendall Tau', fontsize=18)
    elif rdm_comparison=='pearson':
        ax.set_ylabel('Pearson Correlation', fontsize=18)
    elif rdm_comparison=='spearman':
        ax.set_ylabel('Spearman Rho', fontsize=18)

