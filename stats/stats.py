#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 22:29:12 2019

@author: heiko
Functions to check the statistical integrity of the toolbox
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import scipy.signal as signal
from hrf import spm_hrf
from scipy.ndimage import gaussian_filter as gaussian_filter
import pyrsa
import nn_simulations as dnn


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


def check_compare_to_zero(model, n_voxel=100, n_subj=10, n_sim=1000,
                          method='corr', bootstrap='pattern'):
    """ runs simulations for comparison to zero
    It compares whatever model you pass to pure noise data, generated
    as independent normal noise for the voxels and subjects.
    
    Args:
        model(pyrsa.model.Model): the model to be tested against
        n_voxel(int): number of voxels to be simulated per subject
        n_subj(int): number of subjects to be siulated
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
        if bootstrap == 'pattern':
            results = pyrsa.inference.eval_bootstrap_pattern(model, rdms,
                                                             method=method)
        elif bootstrap == 'rdm':
            results = pyrsa.inference.eval_bootstrap_rdm(model, rdms,
                                                         method=method)
        elif bootstrap == 'boot':
            results = pyrsa.inference.eval_bootstrap(model, rdms,
                                                     method=method)
        elif bootstrap == 'crossval':
            results = pyrsa.inference.bootstrap_crossval(model, rdms,
                                                         method=method)
        elif bootstrap == 'crossval_pattern':
            results = pyrsa.inference.bootstrap_crossval(model, rdms,
                                                         method=method,
                                                         k_rdm=1)
        elif bootstrap == 'crossval_rdms':
            results = pyrsa.inference.bootstrap_crossval(model, rdms,
                                                         method=method,
                                                         k_pattern=1)
        idx_valid = ~np.isnan(results.evaluations)
        p[i_sim] = np.sum(results.evaluations[idx_valid] < 1) \
            / np.sum(idx_valid)
    return p


def sampling_DNN(stimuli, n_subj=3, n_vox=100, n_repeat=5, shrinkage=0,
                 model=None, layer=3, sd=np.array([5, 5]),
                 sigma_noise=None, resolution=None, ar_coeff=None,
                 repeats=1, duration=5, pause=1, endzeros=20,
                 cross_residuals=True, sigmaKestimator='eye',
                 sigmaRestimator='eye', use_cor_noise=True,
                 get_cov_estimate=False):
    """ sampling from a neural network layer
    shrinkage governs how strongly sigmaP is shrunk, 0 is raw estimate,
    1 is only diagonal, np.inf is eye
    Noise estimation choices:
        generating noise:
            - on representation-> correlated noise over voxels
    """
    if model is None:
        model = dnn.get_default_model()
    print('\n getting true rdm\n')
    rdm_true = dnn.get_true_rdm(model, layer, stimuli)
    rdm_true_subj = []
    rdm_samples = []
    covs = []
    total_dur = len(stimuli) * repeats * (duration + pause) + endzeros
    print('\n starting simulations')
    for i_subj in tqdm.trange(n_subj):
        Usubj, sigmaP, indices_space, weights = \
            dnn.get_sampled_representations(
                model, layer, sd, stimuli, n_vox)
        data_clean  = pyrsa.data.Dataset(Usubj)
        rdm_true_subj.append(pyrsa.rdm.calc_rdm(data_clean, noise=sigmaP))
        # replaced with proper sampling of a timecourse
        #Usamp = dnn.get_random_sample(Usubj,sigmaP,sigmaNoise,N)
        rdm_samples_subj = np.zeros((len(stimuli), len(stimuli)))
        betas = np.zeros((n_repeat, len(stimuli) + 1, n_vox))
        timecourses = np.zeros((n_repeat, total_dur, n_vox))
        designs = np.zeros((n_repeat, total_dur, len(stimuli) + 1))
        for iSamp in range(n_repeat):
            design = dnn.generate_design_random(len(stimuli), repeats=1,
                duration=duration, pause=pause, endzeros=endzeros)
            if use_cor_noise:
                timecourse = dnn.generate_timecourse(design, Usubj,
                    sigma_noise, resolution=resolution, ar_coeff=ar_coeff,
                    sigmaP=sigmaP)
            else:
                timecourse = dnn.generate_timecourse(design, Usubj,
                    sigma_noise, resolution=resolution, ar_coeff=ar_coeff,
                    sigmaP=None)
            betas[iSamp] = estimate_betas(design, timecourse)
            timecourses[iSamp] = timecourse
            designs[iSamp] = design
        if cross_residuals:
            residuals = get_residuals_cross(designs, timecourses, betas)
        else:
            residuals = get_residuals(designs, timecourses, betas)
        if shrinkage == 0:
            sigma_p_est = np.einsum('ijk,ijl->kl', residuals, residuals) \
                / residuals.shape[0] / residuals.shape[1]
        elif shrinkage <= 1:
            sigma_p_est = np.einsum('ijk,ijl->kl', residuals, residuals) \
                / residuals.shape[0] / residuals.shape[1]
            sigma_p_est = shrinkage * np.diag(np.diag(sigma_p_est)) \
                + (1 - shrinkage) * sigma_p_est
        elif shrinkage == np.inf:
            sigma_p_est = np.eye(n_vox)
        data_samples = [pyrsa.data.Dataset(betas[i, :len(stimuli)])
                        for i in range(len(betas))]
        rdm_samples_subj = pyrsa.rdm.calc_rdm(data_samples, noise=sigma_p_est)
        rdm_samples.append(rdm_samples_subj)
    return rdm_true, rdm_true_subj, rdm_samples, covs


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

