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
import pathlib
from matplotlib.ticker import FormatStrFormatter
import pyrsa
import nn_simulations as dnn
import PIL


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
                          method='corr', bootstrap='pattern',
                          sigma_noise=1):
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
    for i_sim in tqdm.trange(n_sim, position=0):
        raw_u = sigma_noise * np.random.randn(n_subj, n_cond, n_voxel)
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
                         folder='comp_zero', sigma_noise=1):
    """ saves the results of a simulation to a file 
    """
    if not os.path.isdir(folder):
        os.mkdir(folder)
    fname = folder + os.path.sep + 'p_%s_%s_%d_%d_%d_%.2f_%03d.npy' % (method,
        bootstrap, n_cond, n_subj, n_voxel, sigma_noise, idx)
    model_u = np.random.randn(n_cond, n_voxel)
    model_dat = pyrsa.data.Dataset(model_u)
    model_rdm = pyrsa.rdm.calc_rdm(model_dat)
    model = pyrsa.model.ModelFixed('test', model_rdm)
    p = check_compare_to_zero(model, n_voxel=n_voxel, n_subj=n_subj,
                              method=method, bootstrap=bootstrap,
                              sigma_noise=sigma_noise)
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
        bootstrap(String): type of bootstrapping to be performed
            see run_inference for details
        sigma_noise(float): standard deviation of the noise added to the
            representation

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
    D = squareform(target_rdm)
    H = pyrsa.util.matrix.centering(D.shape[0])
    G = -0.5 * (H @ D @ H)
    U0 = pyrsa.simulation.make_signal(G, n_voxel, make_exact=True)
    dat0 = pyrsa.data.Dataset(U0)
    rdm0 = pyrsa.rdm.calc_rdm(dat0)
    p = np.empty(n_sim)
    for i_sim in tqdm.trange(n_sim, position=0):
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
                        folder='comp_model', sigma_noise=1):
    """ saves the results of a simulation to a file 
    """
    if not os.path.isdir(folder):
        os.mkdir(folder)
    fname = folder + os.path.sep + 'p_%s_%s_%d_%d_%d_%.2f_%03d.npy' % (method,
        bootstrap, n_cond, n_subj, n_voxel, sigma_noise, idx)
    model1_u = np.random.randn(n_cond, n_voxel)
    model1_dat = pyrsa.data.Dataset(model1_u)
    model1_rdm = pyrsa.rdm.calc_rdm(model1_dat)
    model1 = pyrsa.model.ModelFixed('test1', model1_rdm)
    model2_u = np.random.randn(n_cond, n_voxel)
    model2_dat = pyrsa.data.Dataset(model2_u)
    model2_rdm = pyrsa.rdm.calc_rdm(model2_dat)
    model2 = pyrsa.model.ModelFixed('test2', model2_rdm)
    p = check_compare_models(model1, model2, n_voxel=n_voxel, n_subj=n_subj,
                              method=method, bootstrap=bootstrap,
                              sigma_noise=sigma_noise)
    np.save(fname, p)


def check_noise_ceiling(model, n_voxel=100, n_subj=10, n_sim=1000,
                         method='corr', bootstrap='pattern', sigma_noise=1,
                         boot_noise_ceil=False):
    """ runs simulations for comparing the model to data generated with the
    model rdm as ground truth to check 
    
    Args:
        model(pyrsa.model.Model): the model to be tested against each other
        n_voxel(int): number of voxels to be simulated per subject
        n_subj(int): number of subjects to be simulated
        n_sim(int): number of simulations to be performed
        bootstrap(String): type of bootstrapping to be performed
            see run_inference for details
        sigma_noise(float): standard deviation of the noise added to the
            representation
        boot_noise_ceil(bool): Whether the noise ceiling is the average
            over bootstrap samples or the evaluation on the original data

    """
    n_cond = int(model.n_cond)
    rdm = model.predict()
    D = squareform(rdm)
    H = pyrsa.util.matrix.centering(D.shape[0])
    G = -0.5 * (H @ D @ H)
    U0 = pyrsa.simulation.make_signal(G, n_voxel, make_exact=True)
    #dat0 = pyrsa.data.Dataset(U0)
    #rdm0 = pyrsa.rdm.calc_rdm(dat0)
    p_upper = np.empty(n_sim)
    p_lower = np.empty(n_sim)
    for i_sim in tqdm.trange(n_sim, position=0):
        raw_u = U0 + sigma_noise * np.random.randn(n_subj, n_cond, n_voxel)
        data = []
        for i_subj in range(n_subj):
            dat = pyrsa.data.Dataset(raw_u[i_subj])
            data.append(dat)
        rdms = pyrsa.rdm.calc_rdm(data)
        results = run_inference(model, rdms, method, bootstrap,
                                boot_noise_ceil=boot_noise_ceil)
        idx_valid = ~np.isnan(results.evaluations[:, 0])
        if boot_noise_ceil:
            p_upper[i_sim] = (np.sum(results.evaluations[idx_valid, 0] <
                                     results.noise_ceiling[1][idx_valid])
                              / np.sum(idx_valid))
            p_lower[i_sim] = (np.sum(results.evaluations[idx_valid, 0] < 
                                     results.noise_ceiling[0][idx_valid])
                              / np.sum(idx_valid))
        else:
            p_upper[i_sim] = (np.sum(results.evaluations[idx_valid, 0] < 
                                     results.noise_ceiling[1])
                              / np.sum(idx_valid))
            p_lower[i_sim] = (np.sum(results.evaluations[idx_valid, 0] < 
                                     results.noise_ceiling[0])
                              / np.sum(idx_valid))
    return np.array([p_lower, p_upper])


def save_noise_ceiling(idx, n_voxel=100, n_subj=10, n_cond=5,
                        method='corr', bootstrap='pattern', sigma_noise=1,
                        folder='comp_noise', boot_noise_ceil=False):
    """ saves the results of a simulation to a file 
    """
    if not os.path.isdir(folder):
        os.mkdir(folder)
    fname = folder + os.path.sep + 'p_%s_%s_%s_%d_%d_%d_%.2f_%03d.npy' % (method,
        bootstrap, boot_noise_ceil, n_cond, n_subj, n_voxel, sigma_noise, idx)
    model_u = np.random.randn(n_cond, n_voxel)
    model_dat = pyrsa.data.Dataset(model_u)
    model_rdm = pyrsa.rdm.calc_rdm(model_dat)
    model = pyrsa.model.ModelFixed('test1', model_rdm)
    p = check_noise_ceiling(model, n_voxel=n_voxel, n_subj=n_subj,
                            method=method, bootstrap=bootstrap,
                            boot_noise_ceil=boot_noise_ceil)
    np.save(fname, p)


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
        if split[2] == 'boot':
            boot = 0
        elif split[2] == 'rdm':
            boot = 1
        elif split[2] == 'pattern':
            boot = 2
        if folder == 'comp_noise':
            if split[3] == 'False':
                boot_noise_ceil = False
            else:
                boot_noise_ceil = True
            ps = ps[0]
        else:
            boot_noise_ceil = False
        n_cond = int(split[-5])
        n_subj = int(split[-4])
        n_voxel = int(split[-3])
        sigma_noise = float(split[-2])
        idx = int(split[-1][:-4])
        desc = np.array([[method, boot, n_subj, n_cond, n_voxel,
                          boot_noise_ceil, sigma_noise, idx]])
        desc = np.repeat(desc, len(ps), axis=0)
        new_ps = np.concatenate((np.array([ps]).T, desc), axis=1)
        table.append(new_ps)
    table = np.concatenate(table, axis=0)
    return table


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


def save_sim_ecoset(model=dnn.get_default_model(), layer=2, sd=0.05,
                    n_voxel=100, n_subj=10, n_stim=92, n_repeat=2,
                    simulation_folder='sim_eco', n_sim=1000,
                    duration=1, pause=1, endzeros=25,
                    use_cor_noise=True, resolution=2,
                    sigma_noise=2, ar_coeff=0.5,
                    ecoset_path='~/ecoset/val/', variation=None):
    """ simulates representations based on randomly chosen ecoset images
        (or any other folder of folders with images inside)
    """
    ecoset_path = pathlib.Path(ecoset_path).expanduser()
    fname_base = get_fname_base(simulation_folder=simulation_folder,
                                layer=layer, n_voxel=n_voxel, n_subj=n_subj,
                                n_repeat=n_repeat, sd=sd, duration=duration,
                                pause=pause, endzeros=endzeros,
                                use_cor_noise=use_cor_noise,
                                resolution=resolution,
                                sigma_noise=sigma_noise,
                                ar_coeff=ar_coeff, variation=variation)
    if not os.path.isdir(fname_base):
        os.makedirs(fname_base)
    for i in tqdm.trange(n_sim):
        # get new stimulus list
        if i == 0 or variation in ['stim', 'both']:
            stim_list = []
            stim_paths = []
            U_complete = []
            folders = os.listdir(ecoset_path)
            for i_stim in range(n_stim):
                i_folder = np.random.randint(len(folders))
                images = os.listdir(os.path.join(ecoset_path,
                                                 folders[i_folder]))
                i_image = np.random.randint(len(images))
                im = PIL.Image.open(os.path.join(ecoset_path,
                                                 folders[i_folder],
                                                 images[i_image]))
                stim_list.append(im)
                stim_paths.append(os.path.join(folders[i_folder],
                                               images[i_image]))
                U_complete.append(
                    dnn.get_complete_representation(model=model, layer=layer,
                                                    stimulus=im))
        U_shape = np.array(U_complete[0].shape)

        # get new sampling locations
        if i == 0 or variation in ['subj', 'both']:
            sigmaP = []
            indices_space = []
            weights = []
            for i_subj in range(n_subj):
                indices_space_subj, weights_subj = dnn.get_random_indices_conv(
                    U_shape, n_voxel)
                sigmaP_subj = dnn.get_sampled_sigmaP(U_shape,
                    indices_space_subj, weights_subj, [sd, sd])
                sigmaP.append(sigmaP_subj)
                indices_space.append(indices_space_subj)
                weights.append(weights_subj)
            sigmaP = np.array(sigmaP)
            indices_space = np.array(indices_space)
            weights = np.array(weights)

        # extract new dnn activations
        if i == 0 or variation in ['subj', 'stim', 'both']:
            Utrue = []
            for i_subj in range(n_subj):
                Utrue_subj = [dnn.sample_representation(np.squeeze(U_c),
                                                        indices_space_subj,
                                                        weights_subj,
                                                        [sd, sd])
                              for U_c in U_complete]
                Utrue_subj = np.array(Utrue_subj)
                Utrue_subj = Utrue_subj / np.sqrt(np.sum(Utrue_subj ** 2)) \
                    * np.sqrt(Utrue_subj.size)
                Utrue.append(Utrue_subj)
            Utrue = np.array(Utrue)

        # run the fmri simulation
        U = []
        des = []
        residuals = []
        for i_subj in range(n_subj):
            timecourses = []
            Usamps = []
            res_subj = []
            for iSamp in range(n_repeat):
                design = dnn.generate_design_random(len(stim_list),
                    repeats=1, duration=duration, pause=pause,
                    endzeros=endzeros)
                if use_cor_noise:
                    timecourse = dnn.generate_timecourse(design, Utrue[i_subj],
                        sigma_noise, resolution=resolution, ar_coeff=ar_coeff,
                        sigmaP=sigmaP[i_subj])
                else:
                    timecourse = dnn.generate_timecourse(design, Utrue_subj,
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
        des = np.array(des)
        np.save(fname_base + 'Utrue%04d' % i, Utrue)
        np.save(fname_base + 'sigmaP%04d' % i, sigmaP)
        np.save(fname_base + 'residuals%04d' % i, residuals)
        np.save(fname_base + 'indices_space%04d' % i, indices_space)
        np.save(fname_base + 'weights%04d' % i, weights)
        np.save(fname_base + 'U%04d' % i, U)
        with open(fname_base + 'stim%04d.txt' % i, 'w') as f:
            for item in stim_paths:
                f.write("%s\n" % item)


def save_simulated_data_dnn(model=dnn.get_default_model(), layer=2, sd=0.05,
                            stim_list=get_stimuli_96(), n_voxel=100, n_subj=10,
                            simulation_folder='sim', n_sim=1000, n_repeat=2,
                            duration=1, pause=1, endzeros=25,
                            use_cor_noise=True, resolution=2,
                            sigma_noise=2, ar_coeff=0.5):
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
                                                stim_list, n_voxel)
            Utrue_subj = Utrue_subj / np.sqrt(np.sum(Utrue_subj ** 2)) \
                * np.sqrt(Utrue_subj.size)
            designs = []
            timecourses = []
            Usamps = []
            res_subj = []
            for iSamp in range(n_repeat):
                design = dnn.generate_design_random(len(stim_list),
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


def analyse_saved_dnn(layer=2, sd=0.05, n_voxel=100, n_repeat=2,
                      n_subj=10, simulation_folder='sim', n_sim=100,
                      duration=1, pause=1, endzeros=25, use_cor_noise=True,
                      resolution=2, sigma_noise=2, ar_coeff=0.5,
                      model_type='fixed_averagetrue',
                      rdm_comparison='cosine', n_Layer=12, k_pattern=3,
                      k_rdm=3, rdm_type='crossnobis', n_stimuli=92,
                      noise_type='eye'):
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
    pat_desc = {'stim': np.arange(n_stimuli)}
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
            for i_smooth in range(len(smoothings)):
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
            for i_smooth in range(len(smoothings)):
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
            for i_smooth in range(len(smoothings)):
                rdm = dnn.get_true_RDM(
                    model=dnn.get_default_model(),
                    layer=i_layer,
                    stimuli=stimuli,
                    smoothing=smoothings[i_smooth],
                    average=True)
                rdm.pattern_descriptors = pat_desc
                rdms.append(rdm)
            for i_smooth in range(len(smoothings)-1,-1,-1):
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


def get_fname_base(simulation_folder, layer, n_voxel, n_subj, n_repeat, sd,
                   duration, pause, endzeros, use_cor_noise, resolution,
                   sigma_noise, ar_coeff, variation=None):
    """ generates the filename base from parameters """
    if variation:
        fname_base = simulation_folder + ('/layer%02d' % layer) \
            + ('/pars_%03d_%02d_%02d_%.3f_%s/' % (
                n_voxel, n_subj, n_repeat, sd, variation)) \
            + ('fmri_%02d_%02d_%03d_%s_%d_%.2f_%.2f/' % (
                duration, pause, endzeros, use_cor_noise, resolution,
                sigma_noise, ar_coeff))
    else:
        fname_base = simulation_folder + ('/layer%02d' % layer) \
            + ('/pars_%03d_%02d_%02d_%.3f/' % (
                n_voxel, n_subj, n_repeat, sd)) \
            + ('fmri_%02d_%02d_%03d_%s_%d_%.2f_%.2f/' % (
                duration, pause, endzeros, use_cor_noise, resolution,
                sigma_noise, ar_coeff))
    return fname_base


def plot_saved_dnn_average(layer=2, sd=3, stim_list=get_stimuli_96(),
                           n_voxel=100, n_subj=10, simulation_folder='test',
                           n_sim=100, n_repeat=2, duration=5, pause=1,
                           endzeros=25, use_cor_noise=True, resolution = 2,
                           sigma_noise=2, ar_coeff=.5, modelType='fixed',
                           model_rdm='averagetrue', n_stimuli=92,
                           rdm_comparison='cosine', n_Layer=12, n_fold=5,
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


def plot_comp(data, alpha=0.05):
    """ plots comp check data
    
    """
    methods = np.unique(data[:, 1])
    boots = np.unique(data[:, 2])
    n_subj = np.unique(data[:, 3])
    n_cond = np.unique(data[:, 4])
    n_voxel = np.unique(data[:, 5])
    boot_noise = np.unique(data[:, 6])
    sigmas = np.unique(data[:, 7])
    idx = np.unique(data[:, 8])
    props = np.nan * np.empty((len(boots), len(n_subj), len(n_cond),
                               len(n_voxel), len(idx)))
    for i_boot in range(len(boots)):
        for i_subj in range(len(n_subj)):
            for i_cond in range(len(n_cond)):
                for i_vox in range(len(n_voxel)):
                    for i in range(len(idx)):
                        dat = data[data[:,2]==boots[i_boot], :]
                        dat = dat[dat[:,3]==n_subj[i_subj], :]
                        dat = dat[dat[:,4]==n_cond[i_cond], :]
                        dat = dat[dat[:,5]==n_voxel[i_vox], :]
                        dat = dat[dat[:,8]==idx[i], :]
                        if len(dat) > 0:
                            prop = (np.sum(dat[:, 0] > (1 - alpha)) 
                                    / len(dat))
                            props[i_boot, i_subj, i_cond, i_vox, i] = prop
                        else:
                            props[i_boot, i_subj, i_cond, i_vox, i] = np.nan
    # First plot: barplot + scatter for each type of bootstrap
    plt.figure()
    ax = plt.subplot(1,1,1)
    for i in range(len(boots)):
        plt.bar(i, np.mean(props[i]))
        plt.plot(np.repeat(i,props[i].size) 
                 + 0.1* np.random.randn(props[i].size),
                 props[i].flatten(), 'k.')
    plt.plot([-0.5,2.5],[alpha, alpha], 'k--')
    plt.xticks([0,1,2], ['both', 'rdm', 'pattern'])
    plt.ylabel('Proportion significant')
    plt.xlabel('bootstrap method')
    # Second plot: plot against n_subj
    p_max = np.nanmax(props)
    plt.figure(figsize=(12,5))
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
        if i==0:
            plt.title('both', fontsize=18)
            plt.ylabel('Proportion significant', fontsize=18)
        elif i==1:
            plt.title('rdm', fontsize=18)
        elif i==2:
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
        plt.plot([-1, len(n_subj)],[alpha, alpha],'k--')
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        plt.xlabel('# of rdms', fontsize=18)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    # Third plot: plot against n_pattern
    plt.figure(figsize=(12,5))
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
        if i==0:
            plt.title('both', fontsize=18)
            plt.ylabel('Proportion significant', fontsize=18)
        elif i==1:
            plt.title('rdm', fontsize=18)
        elif i==2:
            plt.title('pattern', fontsize=18)
            plt.legend([h0[0], h1[0], h2[0], h3[0]], n_subj.astype('int'),
                       frameon=False, title='# of rdms')
        plt.xticks(np.arange(len(n_cond)), n_cond.astype('int'))
        plt.yticks([0, alpha, 2*alpha, 3*alpha])
        plt.ylim([0, p_max + 0.01])
        plt.xlim([-1, len(n_cond)])
        plt.plot([-1, len(n_cond)],[alpha, alpha],'k--')
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        plt.xlabel('# of patterns', fontsize=18)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)


def run_comp(idx):
    """ master script for running the abstract simulations. Each call to 
    this script will run one repetition of the comparisons, i.e. 1000
    evaluations.
    run this script with all indices from 1 to 960 to reproduce  all analyses 
    of this type.
    """
    n_subj = [5,10,20,40]
    n_cond = [5,20,80,160]
    boot_type = ['boot', 'pattern', 'rdm']
    comp_type = ['noise', 'noise_boot', 'model', 'zero']
    n_rep = 5
    (i_rep, i_sub, i_cond, i_boot, i_comp) = np.unravel_index(idx,
        [n_rep, len(n_subj), len(n_cond), len(boot_type), len(comp_type)])
    print('starting simulation:')
    print('%d subjects' % n_subj[i_sub])
    print('%d conditions' % n_cond[i_cond])
    print(boot_type[i_boot])
    print(comp_type[i_comp])
    if i_comp == 0:
        save_noise_ceiling(i_rep, n_subj=n_subj[i_sub], n_cond=n_cond[i_cond],
                           bootstrap=boot_type[i_boot], boot_noise_ceil=False)
    elif i_comp == 1:
        save_noise_ceiling(i_rep, n_subj=n_subj[i_sub], n_cond=n_cond[i_cond],
                           bootstrap=boot_type[i_boot], boot_noise_ceil=True)
    elif i_comp == 2:
        save_compare_models(i_rep, n_subj=n_subj[i_sub], n_cond=n_cond[i_cond],
                           bootstrap=boot_type[i_boot])
    elif i_comp == 3:
        save_compare_to_zero(i_rep, n_subj=n_subj[i_sub], n_cond=n_cond[i_cond],
                           bootstrap=boot_type[i_boot])


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--sim",
                        help="simulation type",
                        choices=['comp'], default='comp')
    parser.add_argument("index", type=int,
                        help="which simulation index to run")
    args = parser.parse_args()
    if args.sim == 'comp':
        run_comp(args.index)

