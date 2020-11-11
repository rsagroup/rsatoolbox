#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions to check the statistical integrity of the toolbox
"""

import os
import pathlib
import numpy as np
from scipy.spatial.distance import squareform
import tqdm
import time
import scipy.signal as signal
import pandas as pd
import sys
import glob
import pyrsa
import nn_simulations as dnn
from hrf import spm_hrf
from helpers import get_fname_base
from helpers import get_resname
from helpers import get_stimuli_ecoset
from helpers import run_inference
from helpers import parse_fmri
from helpers import parse_pars
from helpers import parse_results
from helpers import load_comp
from models import get_models


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


def check_compare_to_zero(model, n_voxel=200, n_subj=10, n_sim=100,
                          method='corr', bootstrap='pattern',
                          sigma_noise=1, test_type='perc'):
    """ runs simulations for comparison to zero
    It compares whatever model you pass to pure noise data, generated
    as independent normal noise for the voxels and subjects.

    Args:
        model(pyrsa.model.Model): the model to be tested against
        n_voxel(int): number of voxels to be simulated per subject
        n_subj(int): number of subjects to be simulated
        n_sim(int): number of simulations to be performed
        test_type(String): test tyoe performed
            'perc' : percentile method, i,e. direct evaluation of samples
            't' : t-test

    """
    n_cond = int(model.n_cond)
    p = np.empty(n_sim)
    for i_sim in tqdm.trange(n_sim, position=1):
        raw_u = sigma_noise * np.random.randn(n_subj, n_cond, n_voxel)
        data = []
        for i_subj in range(n_subj):
            dat = pyrsa.data.Dataset(raw_u[i_subj])
            data.append(dat)
        rdms = pyrsa.rdm.calc_rdm(data)
        results = run_inference(model, rdms, method, bootstrap)
        idx_valid = ~np.isnan(results.evaluations)
        if test_type == 'perc':
            p[i_sim] = 1 - np.sum(results.evaluations[idx_valid] > 0) \
                / np.sum(idx_valid)
        elif test_type == 't':
            p[i_sim] = pyrsa.util.inference_util.t_test_0(
                results.evaluations,
                results.variances,
                dof=results.dof)
        elif test_type == 'ranksum':
            p[i_sim] = pyrsa.util.inference_util.ranksum_value_test(
                results.evaluations)
    return p


def save_compare_to_zero(idx, n_voxel=200, n_subj=10, n_cond=5,
                         method='corr', bootstrap='pattern',
                         folder='comp_zero', sigma_noise=1,
                         test_type='t', n_sim=100):
    """ saves the results of a simulation to a file
    """
    if not os.path.isdir(folder):
        os.mkdir(folder)
    fname = folder + os.path.sep + 'p_%s_%s_%s_%d_%d_%d_%.2f_%03d.npy' % (
        method, bootstrap, test_type,
        n_cond, n_subj, n_voxel, sigma_noise, idx)
    if not os.path.isfile(fname):
        model_u = np.random.randn(n_cond, n_voxel)
        model_dat = pyrsa.data.Dataset(model_u)
        model_rdm = pyrsa.rdm.calc_rdm(model_dat)
        model = pyrsa.model.ModelFixed('test', model_rdm)
        p = check_compare_to_zero(model, n_voxel=n_voxel, n_subj=n_subj,
                                  method=method, bootstrap=bootstrap,
                                  sigma_noise=sigma_noise,
                                  test_type=test_type, n_sim=n_sim)
        np.save(fname, p)


def check_compare_models(model1, model2, n_voxel=200, n_subj=10, n_sim=100,
                         method='corr', bootstrap='pattern', sigma_noise=1,
                         test_type='t'):
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
    target_rdm = target_rdm + np.max(target_rdm) + 0.01
    D = squareform(target_rdm)
    H = pyrsa.util.matrix.centering(D.shape[0])
    G = -0.5 * (H @ D @ H)
    G = G + np.eye(G.shape[0])
    U0 = pyrsa.simulation.make_signal(G, n_voxel, make_exact=True)
    # dat0 = pyrsa.data.Dataset(U0)
    # rdm0 = pyrsa.rdm.calc_rdm(dat0)
    p = np.empty(n_sim)
    for i_sim in tqdm.trange(n_sim, position=0):
        raw_u = U0 + sigma_noise * np.random.randn(n_subj, n_cond, n_voxel)
        data = []
        for i_subj in range(n_subj):
            dat = pyrsa.data.Dataset(raw_u[i_subj])
            data.append(dat)
        rdms = pyrsa.rdm.calc_rdm(data)
        results = run_inference([model1, model2], rdms, method, bootstrap)
        if test_type == 'perc':
            idx_valid = ~np.isnan(results.evaluations[:, 0])
            p = (np.sum(results.evaluations[idx_valid, 0] >
                        results.evaluations[idx_valid, 1])
                 / np.sum(idx_valid))
            p[i_sim] = 2 * np.min(p, 1 - p)
        elif test_type == 't':
            p[i_sim] = pyrsa.util.inference_util.t_tests(
                results.evaluations, results.variances, results.dof)[0, 1]
        elif test_type == 'ranksum':
            p[i_sim] = pyrsa.util.inference_util.ranksum_pair_test(
                results.evaluations)[0, 1]
    return p


def save_compare_models(idx, n_voxel=200, n_subj=10, n_cond=5,
                        method='corr', bootstrap='pattern',
                        folder='comp_model', sigma_noise=1,
                        test_type='t', n_sim=100):
    """ saves the results of a simulation to a file
    """
    if not os.path.isdir(folder):
        os.mkdir(folder)
    fname = folder + os.path.sep + 'p_%s_%s_%s_%d_%d_%d_%.2f_%03d.npy' % (
        method, bootstrap, test_type,
        n_cond, n_subj, n_voxel, sigma_noise, idx)
    if not os.path.isfile(fname):
        model1_u = np.random.randn(n_cond, n_voxel)
        model1_dat = pyrsa.data.Dataset(model1_u)
        model1_rdm = pyrsa.rdm.calc_rdm(model1_dat)
        model1 = pyrsa.model.ModelFixed('test1', model1_rdm)
        model2_u = np.random.randn(n_cond, n_voxel)
        model2_dat = pyrsa.data.Dataset(model2_u)
        model2_rdm = pyrsa.rdm.calc_rdm(model2_dat)
        model2 = pyrsa.model.ModelFixed('test2', model2_rdm)
        p = check_compare_models(model1, model2, n_voxel=n_voxel,
                                 n_subj=n_subj,
                                 method=method, bootstrap=bootstrap,
                                 sigma_noise=sigma_noise,
                                 n_sim=n_sim)
        np.save(fname, p)


def check_noise_ceiling(model, n_voxel=200, n_subj=10, n_sim=100,
                        method='corr', bootstrap='pattern', sigma_noise=1,
                        boot_noise_ceil=False, test_type='t'):
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
        test_type(String): How to perform the test
            't' : t-test
            'perc' : bootstrap percentiles

    """
    n_cond = int(model.n_cond)
    rdm = model.predict()
    D = squareform(rdm)
    H = pyrsa.util.matrix.centering(D.shape[0])
    G = -0.5 * (H @ D @ H)
    U0 = pyrsa.simulation.make_signal(G, n_voxel, make_exact=True)
    # dat0 = pyrsa.data.Dataset(U0)
    # rdm0 = pyrsa.rdm.calc_rdm(dat0)
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
        if test_type == 'perc':
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
        elif test_type == 't':
            if results.noise_ceil_var is None:
                p_upper[i_sim] = pyrsa.util.inference_util.t_test_nc(
                    results.evaluations, results.variances,
                    results.noise_ceiling[1], None,
                    results.dof)
                p_lower[i_sim] = pyrsa.util.inference_util.t_test_nc(
                    results.evaluations, results.variances,
                    results.noise_ceiling[0], None,
                    results.dof)
            else:
                p_upper[i_sim] = pyrsa.util.inference_util.t_test_nc(
                    results.evaluations, results.variances,
                    np.nanmean(results.noise_ceiling[1]),
                    results.noise_ceil_var[:-2, 1],
                    results.dof)
                p_lower[i_sim] = pyrsa.util.inference_util.t_test_nc(
                    results.evaluations, results.variances,
                    np.nanmean(results.noise_ceiling[0]),
                    results.noise_ceil_var[:-1, 0],
                    results.dof)
        elif test_type == 'ranksum':
            p_upper[i_sim] = pyrsa.util.inference_util.ranksum_value_test(
                results.evaluations,
                comp_value=np.mean(results.noise_ceiling[1]))
            p_lower[i_sim] = pyrsa.util.inference_util.ranksum_value_test(
                results.evaluations,
                comp_value=np.mean(results.noise_ceiling[0]))
    return np.array([p_lower, p_upper])


def save_noise_ceiling(idx, n_voxel=200, n_subj=10, n_cond=5,
                       method='corr', bootstrap='pattern', sigma_noise=1,
                       folder='comp_noise', boot_noise_ceil=False,
                       test_type='t', n_sim=100):
    """ saves the results of a simulation to a file
    """
    if not os.path.isdir(folder):
        os.mkdir(folder)
    fname = (folder + os.path.sep + 'p_%s_%s_%s_%s_%d_%d_%d_%.2f_%03d.npy'
             % (method, bootstrap, test_type, boot_noise_ceil,
                n_cond, n_subj, n_voxel,
                sigma_noise, idx))
    if not os.path.isfile(fname):
        model_u = np.random.randn(n_cond, n_voxel)
        model_dat = pyrsa.data.Dataset(model_u)
        model_rdm = pyrsa.rdm.calc_rdm(model_dat)
        model = pyrsa.model.ModelFixed('test1', model_rdm)
        p = check_noise_ceiling(model, n_voxel=n_voxel, n_subj=n_subj,
                                method=method, bootstrap=bootstrap,
                                boot_noise_ceil=boot_noise_ceil,
                                test_type=test_type, n_sim=n_sim)
        np.save(fname, p)


def sim_ecoset(layer=2, sd=0.05, n_stim_all=320,
               n_voxel=100, n_subj=10, n_stim=40, n_repeat=2,
               simulation_folder='sim_eco', n_sim=100,
               duration=1, pause=1, endzeros=25,
               use_cor_noise=True, resolution=2,
               sigma_noise=1, ar_coeff=0.5,
               ecoset_path='~/ecoset/val/', variation=None,
               model_type='fixed_average',
               rdm_comparison='cosine', n_layer=12, k_pattern=None,
               k_rdm=None, rdm_type='crossnobis',
               noise_type='residuals', boot_type='both',
               start_idx=0, smoothing=None):
    """ simulates representations based on randomly chosen ecoset images
        (or any other folder of folders with images inside)
        and directly runs the analysis on it saving only the subject
        wise information and the results of the analysis.
        -> Minimal disc usage variant
    """
    model = dnn.get_default_model()
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
    res_name = get_resname(boot_type, rdm_type, model_type, rdm_comparison,
                           noise_type, n_stim, k_pattern, k_rdm,
                           smoothing=smoothing)
    res_path = fname_base + res_name
    print(res_path, flush=True)
    if smoothing is None:
        smoothing = sd
    if not os.path.isdir(res_path):
        os.mkdir(res_path)
    for i in tqdm.trange(start_idx, n_sim):
        # get stimulus list or save one if there is none yet
        if os.path.isfile(fname_base + 'stim%04d.txt' % i):
            stim_list = []
            f = open(os.path.join(fname_base, 'stim%04d.txt' % i))
            stim_paths = f.read()
            f.close()
            stim_paths = stim_paths.split('\n')
        elif i == start_idx or variation in ['stim', 'both']:
            stim_paths = []
            folders = os.listdir(ecoset_path)
            for i_stim in range(n_stim_all):
                i_folder = np.random.randint(len(folders))
                images = os.listdir(os.path.join(ecoset_path,
                                                 folders[i_folder]))
                i_image = np.random.randint(len(images))
                stim_paths.append(os.path.join(folders[i_folder],
                                               images[i_image]))
        if not os.path.isfile(fname_base + 'stim%04d.txt' % i):
            with open(fname_base + 'stim%04d.txt' % i, 'w') as f:
                for item in stim_paths:
                    f.write("%s\n" % item)
        stim_list = get_stimuli_ecoset(ecoset_path, stim_paths[:n_stim])
        # Recalculate U_complete if necessary
        if i == start_idx or variation in ['stim', 'both']:
            U_complete = []
            for i_stim in range(n_stim):
                U_complete.append(
                    dnn.get_complete_representation(
                        model=model, layer=layer,
                        stimulus=stim_list[i_stim]))
            U_shape = np.array(U_complete[0].shape)

        # get new sampling locations if necessary
        if os.path.isfile(fname_base + 'weights%04d.npy' % i):
            indices_space = np.load(fname_base + 'indices_space%04d.npy' % i)
            weights = np.load(fname_base + 'weights%04d.npy' % i)
            sigmaP = []
            for i_subj in range(n_subj):
                sigmaP_subj = dnn.get_sampled_sigmaP(
                    U_shape, indices_space[i_subj], weights[i_subj], [sd, sd])
                sigmaP.append(sigmaP_subj)
            sigmaP = np.array(sigmaP)
        elif i == start_idx or variation in ['subj', 'both']:
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
        if not os.path.isfile(fname_base + 'weights%04d.npy' % i):
            np.save(fname_base + 'weights%04d.npy' % i, weights)
        if not os.path.isfile(fname_base + 'indices_space%04d.npy' % i):
            np.save(fname_base + 'indices_space%04d.npy' % i, indices_space)

        # extract new dnn activations
        if i == start_idx or variation in ['subj', 'stim', 'both']:
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
        for i_subj in range(n_subj):
            timecourses = []
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

        # run analysis
        # get models if stimulus changed, subjects are irrelevant
        if i == start_idx or variation in ['stim', 'both']:
            models = get_models(
                model_type, stim_list,
                n_layer=n_layer,
                smoothing=smoothing)
        # calculate RDMs
        data = []
        desc = {'stim': np.tile(np.arange(n_stim), n_repeat),
                'repeat': np.repeat(np.arange(n_repeat), n_stim)}
        for i_subj in range(U.shape[0]):
            u_subj = U[i_subj, :, :n_stim, :].reshape(n_repeat * n_stim,
                                                      n_voxel)
            data.append(pyrsa.data.Dataset(u_subj, obs_descriptors=desc))
        if noise_type == 'eye':
            noise = None
        elif noise_type == 'residuals':
            noise = pyrsa.data.prec_from_residuals(residuals)
        rdms = pyrsa.rdm.calc_rdm(data, method=rdm_type, descriptor='stim',
                                  cv_descriptor='repeat', noise=noise)
        # get true U RDMs
        # dat_true = []
        # for i_subj in range(U.shape[0]):
        #     dat_true.append(pyrsa.data.Dataset(Utrue[i_subj]))
        # rdms_true = pyrsa.rdm.calc_rdm(dat_true)
        # run inference & save it
        results = run_inference(models, rdms, method=rdm_comparison,
                                bootstrap=boot_type,
                                k_rdm=k_rdm, k_pattern=k_pattern)
        results.save(res_path + '/res%04d.hdf5' % (i))


def save_metric(idx, simulation_folder='sim_metric'):
    """ saves simulations for the comparison of metrics, without bootstrapping!
    uses the dnn-ecoset based simulations
    """
    rdm_comparisons = ['cosine', 'spearman', 'corr', 'kendall', 'tau-a',
                       'rho-a', 'corr_cov', 'cosine_cov']
    variations = [None, 'stim', 'subj', 'both']
    i_comp, i_var = np.unravel_index(idx, (8, 4))
    rdm_comparison = rdm_comparisons[i_comp]
    variation = variations[i_var]
    sim_ecoset(layer=8, sd=0.05, n_stim_all=320,
               n_voxel=100, n_subj=10, n_stim=40, n_repeat=2,
               simulation_folder=simulation_folder, n_sim=100,
               duration=1, pause=1, endzeros=25,
               use_cor_noise=True, resolution=2,
               sigma_noise=0.5, ar_coeff=0.5,
               ecoset_path='~/ecoset/val/',
               variation=variation,
               model_type='fixed_average',
               rdm_comparison=rdm_comparison,
               n_layer=12, k_pattern=None,
               k_rdm=None, rdm_type='crossnobis',
               noise_type='residuals', boot_type='fix',
               start_idx=0)


def save_rdm_type(idx, simulation_folder='sim_type'):
    """ saves simulations for the comparison of rdm types
    without bootstrapping!
    uses the dnn-ecoset based simulations
    """
    rdm_types = ['euclidean', 'correlation', 'mahalanobis', 'crossnobis']
    variations = [None, 'stim', 'subj', 'both']
    i_type, i_var = np.unravel_index(idx, (8, 4))
    rdm_type = rdm_types[i_type]
    variation = variations[i_var]
    sim_ecoset(layer=8, sd=0.05, n_stim_all=320,
               n_voxel=100, n_subj=10, n_stim=40, n_repeat=2,
               simulation_folder=simulation_folder, n_sim=100,
               duration=1, pause=1, endzeros=25,
               use_cor_noise=True, resolution=2,
               sigma_noise=0.5, ar_coeff=0.5,
               ecoset_path='~/ecoset/val/',
               variation=variation,
               model_type='fixed_average',
               rdm_comparison='corr',
               n_layer=12, k_pattern=None,
               k_rdm=None, rdm_type=rdm_type,
               noise_type='residuals', boot_type='fix',
               start_idx=0)


def run_flex(idx, start_idx, simulation_folder='sim_flex',
             ecoset_path='~/ecoset/val/'):
    """ runs the flexible model checks
    based on dnn-ecoset simulations
    """
    models = [
        ['fixed_full', 0],
        ['fixed_full', 0.05],
        ['fixed_full', np.inf],
        ['fixed_average', 0],
        ['fixed_average', 0.05],
        ['fixed_average', np.inf],
        ['fixed_mean', 0],
        ['fixed_mean', 0.05],
        ['fixed_mean', np.inf],
        ['select_full', None],
        ['select_mean', None],
        ['select_average', None],
        ['select_both', None],
        ['weighted_avgfull', None],
        ['interpolate_full', None],
        ['interpolate_mean', None],
        ['interpolate_average', None],
        ['interpolate_full', None]]
    
    variation = 'both'
    boot = 'fancy'
    layer = 8
    n_repeat = 4
    n_stim = 40
    n_vox = 100
    n_subj = 20
    sd = 0.05
    sigma_noise = 1
    rdm_type = 'crossnobis'
    model_type = models[idx][0]
    smoothing = models[idx][1]
    rdm_comparison = 'corr'
    noise_type = 'residuals'
    k_pattern = None
    k_rdm = None
    # check how far the simulation was processed
    fname_base = get_fname_base(simulation_folder=simulation_folder,
                                layer=layer, n_voxel=n_vox,
                                n_subj=n_subj,
                                n_repeat=n_repeat,
                                sd=sd,
                                duration=1, pause=1, endzeros=25,
                                resolution=2,
                                ar_coeff=0.5,
                                use_cor_noise=True,
                                sigma_noise=sigma_noise,
                                variation=variation)
    res_name = get_resname(boot, rdm_type, model_type,
                           rdm_comparison, noise_type, n_stim,
                           k_pattern, k_rdm, smoothing)
    fname = 'res%04d.hdf5' % start_idx
    if not os.path.isdir(fname_base):
        os.mkdir(fname_base)
    if not os.path.isdir(os.path.join(fname_base, res_name)):
        os.mkdir(os.path.join(fname_base, res_name))
    if not os.path.isfile(os.path.join(
            fname_base, res_name, fname)):
        sim_ecoset(layer=layer, sd=sd, n_stim_all=80,
            n_voxel=n_vox, n_subj=n_subj, n_stim=n_stim,
            n_repeat=n_repeat,
            simulation_folder=simulation_folder, n_sim=start_idx + 1,
            sigma_noise=sigma_noise,
            ecoset_path=ecoset_path, variation=variation,
            model_type=model_type,
            rdm_comparison=rdm_comparison, n_layer=12,
            k_pattern=k_pattern, k_rdm=k_rdm,
            rdm_type='crossnobis',
            noise_type=noise_type, boot_type=boot,
            start_idx=start_idx,
            smoothing=smoothing)
    print(f'index # {idx}, simulation # {start_idx} complete\n', flush=True)


def fix_flex(simulation_folder='sim_flex', ecoset_path='~/ecoset/val/'):
    """runs single flexible model simulations to allow parallelization
    """
    indices = np.random.permutation(1400)
    for idx in indices:
        model_idx = int(np.floor(idx / 100))
        start_idx = int(idx % 100)
        run_flex(model_idx, start_idx, simulation_folder=simulation_folder,
                 ecoset_path=ecoset_path)


def run_comp(idx):
    """ master script for running the abstract simulations. Each call to
    this script will run one repetition of the comparisons, i.e. 100
    evaluations.
    run this script with all indices from 1 to 960 to reproduce  all analyses
    of this type.
    """
    n_subj = [5, 10, 20, 40]
    n_cond = [5, 20, 80, 160]
    boot_type = [['both', 'perc'],
                 ['pattern', 'perc'],
                 ['rdm', 'perc'],
                 ['fix', 'ranksum'],
                 ['both', 't'],
                 ['pattern', 't'],
                 ['rdm', 't'],
                 ['fix', 't'],
                 ['fancyboot', 't']]
    # comp_type = ['noise', 'noise_boot', 'model', 'zero']
    comp_type = ['model', 'zero']
    n_rep = 50
    (i_boot, i_rep, i_sub, i_cond, i_comp) = np.unravel_index(
        idx,
        [len(boot_type), n_rep, len(n_subj), len(n_cond), len(comp_type)])
    print(idx, flush=True)
    print('starting simulation:', flush=True)
    print('%d subjects' % n_subj[i_sub], flush=True)
    print('%d conditions' % n_cond[i_cond], flush=True)
    print(boot_type[i_boot], flush=True)
    print(comp_type[i_comp], flush=True)
    if i_comp == 2:
        save_noise_ceiling(i_rep, n_subj=n_subj[i_sub], n_cond=n_cond[i_cond],
                           bootstrap=boot_type[i_boot][0],
                           test_type=boot_type[i_boot][1],
                           boot_noise_ceil=False)
    elif i_comp == 3:
        save_noise_ceiling(i_rep, n_subj=n_subj[i_sub], n_cond=n_cond[i_cond],
                           bootstrap=boot_type[i_boot][0],
                           test_type=boot_type[i_boot][1],
                           boot_noise_ceil=True)
    elif i_comp == 0:
        save_compare_models(i_rep, n_subj=n_subj[i_sub], n_cond=n_cond[i_cond],
                            bootstrap=boot_type[i_boot][0],
                            test_type=boot_type[i_boot][1])
    elif i_comp == 1:
        save_compare_to_zero(i_rep, n_subj=n_subj[i_sub],
                             n_cond=n_cond[i_cond],
                             bootstrap=boot_type[i_boot][0],
                             test_type=boot_type[i_boot][1])


def run_eco(idx, ecoset_path=None, start_idx=0):
    """ master script for running the ecoset simulations. Each call to
    this script will run one repetition of the comparisons, i.e. 100
    evaluations.
    run this script with all indices from 1 to 2400 to reproduce all analyses
    of this type.
    """
    if ecoset_path is None:
        ecoset_path = '~/ecoset/val/'
    variation, boot, i_var, layer, i_layer, n_repeat, i_repeat, n_stim, \
        i_stim, n_subj, i_sub, sd, i_sd, n_vox, i_vox, noise_type = \
        _resolve_idx(idx)
    print('starting simulation:', flush=True)
    print('variation: %s' % variation[i_var], flush=True)
    print('layer: %d' % layer[i_layer], flush=True)
    print('%d repeats' % n_repeat[i_repeat], flush=True)
    print('%d stimuli' % n_stim[i_stim], flush=True)
    print('%d subjects' % n_subj[i_sub], flush=True)
    print('%.3f sd' % sd[i_sd], flush=True)
    print('%d voxel' % n_vox[i_vox], flush=True)
    print('\n\n\n\n', flush=True)
    time.sleep(1)
    if variation[i_var][:4] == 'None':
        sim_ecoset(variation=None,
                   layer=layer[i_layer],
                   n_repeat=n_repeat[i_repeat],
                   n_subj=n_subj[i_sub],
                   sd=sd[i_sd], boot_type=boot[i_var],
                   ecoset_path=ecoset_path,
                   rdm_comparison='corr',
                   n_stim=n_stim[i_stim],
                   n_sim=100, n_voxel=n_vox[i_vox],
                   noise_type=noise_type,
                   sigma_noise=1, start_idx=start_idx)
    else:
        sim_ecoset(variation=variation[i_var],
                   layer=layer[i_layer],
                   n_repeat=n_repeat[i_repeat],
                   n_subj=n_subj[i_sub],
                   sd=sd[i_sd], boot_type=boot[i_var],
                   ecoset_path=ecoset_path,
                   rdm_comparison='corr',
                   n_stim=n_stim[i_stim],
                   n_sim=100, n_voxel=n_vox[i_vox],
                   noise_type=noise_type,
                   sigma_noise=1, start_idx=start_idx)


def summarize_eco(simulation_folder='sim_eco'):
    """ collects the existing ecoset simulations and creates a list of
    simulation labels and the means and standard deviations of the bootstrap
    model evaluations.
    Results are saved at the toplevel of the folder next to the layer folders

    Args:
        simulation_folder : folder
            Which folder to go through. The default is 'sim_eco'.

    """
    data_labels = pd.DataFrame({
        'layer': [], 'n_voxel': [], 'n_subj': [], 'n_rep': [],
        'sd': [], 'variation': [], 'duration': [], 'pause': [],
        'endzeros': [], 'use_cor_noise': [], 'resolution': [],
        'sigma_noise': [], 'ar_coeff': [], 'boot_type': [],
        'rdm_type': [], 'model_type': [], 'rdm_comparison': [],
        'noise_type': [], 'n_stim': []})
    means = []
    variances = []
    pairs = []
    for layer in [i for i in os.listdir(simulation_folder)
                  if i[:5] == 'layer']:
        i_layer = int(layer[-2:])
        for pars in os.listdir(os.path.join(simulation_folder, layer)):
            n_voxel, n_subj, n_rep, sd, variation = parse_pars(pars)
            for fmri in os.listdir(
                    os.path.join(simulation_folder, layer, pars)):
                duration, pause, endzeros, use_cor_noise, resolution, \
                    sigma_noise, ar_coeff = parse_fmri(fmri)
                for results in pathlib.Path(
                        os.path.join(simulation_folder, layer, pars, fmri)
                        ).glob('results_*'):
                    sys.stdout.write(str(results) + '\n')
                    res_string = os.path.split(results)[-1]
                    boot_type, rdm_type, model_type, rdm_comparison, \
                        noise_type, n_stim = parse_results(res_string)
                    data_labels = data_labels.append(
                        {'layer': i_layer,
                         'n_voxel': n_voxel, 'n_subj': n_subj,
                         'n_rep': n_rep, 'sd': sd,
                         'variation': variation, 'endzeros': endzeros,
                         'duration': duration, 'pause': pause,
                         'use_cor_noise': use_cor_noise,
                         'resolution': resolution,
                         'sigma_noise': sigma_noise, 'ar_coeff': ar_coeff,
                         'boot_type': boot_type, 'noise_type': noise_type,
                         'rdm_comparison': rdm_comparison,
                         'rdm_type': rdm_type, 'model_type': model_type,
                         'n_stim': n_stim},
                        ignore_index=True)
                    mean = np.nan * np.zeros((100, 12))
                    variance = np.nan * np.zeros((100, 12, 12))
                    pairwise = np.nan * np.zeros((100, 12, 12))
                    for i_res in results.glob('res*.hdf5'):
                        idx = int(str(i_res)[-9:-5])
                        try:
                            res = pyrsa.inference.load_results(
                                i_res, file_type='hdf5')
                            if res.evaluations.ndim == 2:
                                no_nan_idx = ~np.isnan(res.evaluations[:, 0])
                            elif res.evaluations.ndim == 3:
                                no_nan_idx = \
                                    ~np.isnan(res.evaluations[:, 0, 0])
                            if np.any(no_nan_idx):
                                for i in range(12):
                                    for j in range(12):
                                        diff = (res.evaluations[:, i]
                                                - res.evaluations[:, j])
                                        pairwise[idx, i, j] = np.sum(
                                            diff[no_nan_idx] > 0)
                                m = np.mean(res.evaluations[no_nan_idx],
                                            axis=0)
                                while m.ndim > 1:
                                    m = np.mean(m, axis=-1)
                                mean[idx] = m
                                variance[idx] = res.variances
                            else:
                                raise OSError('no valid results')
                        except OSError:
                            mean[idx] = np.nan
                            variance[idx] = np.nan
                            pairwise[idx] = np.nan
                    means.append(mean)
                    variances.append(variance)
                    pairs.append(pairwise)
            means_array = np.array(means)
            vars_array = np.array(variances)
            pairs_array = np.array(pairs)
            np.save(os.path.join(simulation_folder, 'means.npy'), means_array)
            np.save(os.path.join(simulation_folder, 'stds.npy'), vars_array)
            np.save(os.path.join(simulation_folder, 'pairs.npy'), pairs_array)
            data_labels.to_csv(os.path.join(simulation_folder, 'labels.csv'))
    return data_labels, means, variances, pairs


def check_eco(simulation_folder='sim_eco', N=100):
    """ checks which simulations are complete and lists incomplete simulations
    i.e. simulations which were started but not finished
    """
    folders = os.listdir(simulation_folder)
    for folder in folders:
        folder_path = os.path.join(simulation_folder, folder)
        if os.path.isdir(folder_path):
            pars = os.listdir(folder_path)
            for par in pars:
                par_path = os.path.join(folder_path, par)
                fmris = os.listdir(par_path)
                for fmri in fmris:
                    path = os.path.join(par_path, fmri)
                    # check for completeness of stimulus and indices
                    break_results = False
                    for i in range(100):
                        if not os.path.isfile(os.path.join(
                                path, 'stim%04d.txt' % i)):
                            print(path)
                            print('stimuli incomplete')
                            break
                        if not os.path.isfile(os.path.join(
                                path, 'indices_space%04d.npy' % i)):
                            print(path)
                            print('indices incomplete')
                            break
                        for i_res in glob.glob(os.path.join(path, 'results*')):
                            if not os.path.isfile(os.path.join(
                                    i_res, 'res%04d.hdf5' % i)):
                                print(i_res)
                                print('results incomplete')
                                break_results = True
                                break
                        if break_results:
                            break


def fix_eco(
        simulation_folder='sim_eco', n_sim=100,
        duration=1, pause=1, endzeros=25, resolution=2,
        sigma_noise=1, ar_coeff=0.5, ecoset_path='~/ecoset/val/',
        model_type='fixed_average',
        rdm_comparison='corr', n_layer=12, k_pattern=None,
        k_rdm=None, rdm_type='crossnobis',
        boot_type='both'):
    """
    checks the completeness of eco simulations and starts a completion job
    if a simulation is not finished yet. Proceeds in random order
    only ends if all simulations are complete.
    """
    indices = np.random.permutation(2400)
    for idx in indices:
        variation, boot, i_var, layer, i_layer, n_repeat, i_repeat, n_stim, \
            i_stim, n_subj, i_sub, sd, i_sd, n_vox, i_vox, noise_type = \
            _resolve_idx(idx)
        # check how far the simulation was processed
        if variation[i_var][:4] == 'None':
            vari = None
        else:
            vari = variation[i_var]
        fname_base = get_fname_base(simulation_folder=simulation_folder,
                                    layer=layer[i_layer], n_voxel=n_vox[i_vox],
                                    n_subj=n_subj[i_sub],
                                    n_repeat=n_repeat[i_repeat],
                                    sd=sd[i_sd], duration=duration,
                                    pause=pause, endzeros=endzeros,
                                    use_cor_noise=True,
                                    resolution=resolution,
                                    sigma_noise=sigma_noise,
                                    ar_coeff=ar_coeff,
                                    variation=vari)
        if os.path.isdir(fname_base):
            res_name = get_resname(boot[i_var], rdm_type, model_type,
                                   rdm_comparison, noise_type, n_stim[i_stim],
                                   k_pattern, k_rdm)
            if os.path.isdir(os.path.join(fname_base, res_name)):
                start_idx = 0
                for i_res in pathlib.Path(
                        os.path.join(fname_base, res_name)
                        ).glob('*res*.hdf5'):
                    i = int(str(i_res)[-9:-5])
                    if i >= start_idx:
                        start_idx = i + 1
            else:
                start_idx = 0
        else:
            start_idx = 0
        if start_idx < 100:
            run_eco(idx, ecoset_path=ecoset_path, start_idx=start_idx)
        print(f'index # {idx} complete\n', flush=True)


def _resolve_idx(idx):
    """ helper to convert linear index into simulation parameters
    (of eco simulations)

    """
    # combined with all
    variation = ['None_both', 'None_stim', 'None_subj', 'None_fancy',
                 'both', 'stim', 'subj', 'both']
    boot = ['both', 'pattern', 'rdm', 'fancyboot',
            'both', 'pattern', 'rdm', 'fancyboot']
    n_subj = [5, 10, 20, 40, 80]
    n_stim = [10, 20, 40, 80, 160]

    # varied separately
    n_repeat = [4, 2, 8]
    layer = [8, 2, 5, 10, 12]
    sd = [0.05, 0, 0.25, np.inf]
    n_vox = [100, 10, 1000]
    i_separate = int(idx / 200)
    i_all = idx % 200

    (i_stim, i_sub, i_var) = np.unravel_index(
        i_all, [len(n_stim), len(n_subj), len(variation)])
    if i_separate < 5:
        i_layer = i_separate
        i_repeat = 0
        i_sd = 0
        i_vox = 0
    elif i_separate < 7:
        i_layer = 0
        i_repeat = i_separate - 4
        i_sd = 0
        i_vox = 0
    elif i_separate < 10:
        i_layer = 0
        i_repeat = 0
        i_sd = i_separate - 6
        i_vox = 0
    elif i_separate < 12:
        i_layer = 0
        i_repeat = 0
        i_sd = 0
        i_vox = i_separate - 9
    if i_vox == 2:
        noise_type = 'eye'
    else:
        noise_type = 'residuals'
    return (variation, boot, i_var, layer, i_layer, n_repeat, i_repeat, n_stim,
            i_stim, n_subj, i_sub, sd, i_sd, n_vox, i_vox, noise_type)


def boot_cv_sim(i=0, n_cv=2, i_rep=0, ecoset_path='~/ecoset/val/',
                simulation_folder='boot_cv'):
    layer = 8
    n_voxel = 100
    n_subj = 20
    n_stim = 40
    n_stim_all = 40
    n_repeat = 4
    sd = 0.05
    smoothing = sd
    duration = 1
    pause = 1
    endzeros = 25
    use_cor_noise = True
    sigma_noise = 1
    resolution = 2
    ar_coeff = 0.5
    model_type = 'select_average'
    rdm_type = 'crossnobis'
    rdm_comparison = 'corr'
    boot_type = 'crossval'
    
    model = dnn.get_default_model()
    ecoset_path = pathlib.Path(ecoset_path).expanduser()
    res_path = os.path.join(simulation_folder, f'cv_{n_cv}')
    if not os.path.isdir(res_path):
        os.makedirs(res_path)
    res_file = '/res%04d_%03d.hdf5' % (i, i_rep)
    full_path = os.path.join(res_path, res_file)
    if os.path.isfile(full_path):
        print(full_path)
        return
    # get stimulus list or save one if there is none yet
    stim_file = os.path.join(simulation_folder, 'stim%04d.txt' % i)
    if os.path.isfile(stim_file):
        f = open(stim_file, )
        stim_paths = f.read()
        f.close()
        stim_paths = stim_paths.split('\n')
    else:
        stim_paths = []
        folders = os.listdir(ecoset_path)
        for i_stim in range(n_stim_all):
            i_folder = np.random.randint(len(folders))
            images = os.listdir(os.path.join(ecoset_path,
                                             folders[i_folder]))
            i_image = np.random.randint(len(images))
            stim_paths.append(os.path.join(folders[i_folder],
                                           images[i_image]))
    if not os.path.isfile(stim_file):
        with open(stim_file, 'w') as f:
            for item in stim_paths:
                f.write("%s\n" % item)
    stim_list = get_stimuli_ecoset(ecoset_path, stim_paths[:n_stim])
    rdm_file = os.path.join(simulation_folder, 'rdm%04d.hdf5' % i)
    if os.path.isfile(rdm_file):
        rdms = pyrsa.rdm.load_rdm(rdm_file, file_type='hdf5')
    else:
        # Recalculate U_complete if necessary
        U_complete = []
        for i_stim in range(n_stim):
            U_complete.append(
                dnn.get_complete_representation(
                    model=model, layer=layer,
                    stimulus=stim_list[i_stim]))
        U_shape = np.array(U_complete[0].shape)
    
        # get new sampling locations if necessary
        weight_file = os.path.join(simulation_folder,
                                   'weights%04d.npy' % i)
        indices_file = os.path.join(simulation_folder,
                                    'indices_space%04d.npy' % i)
        if os.path.isfile(weight_file):
            indices_space = np.load(indices_file)
            weights = np.load(weight_file)
            sigmaP = []
            for i_subj in range(n_subj):
                sigmaP_subj = dnn.get_sampled_sigmaP(
                    U_shape, indices_space[i_subj], weights[i_subj], [sd, sd])
                sigmaP.append(sigmaP_subj)
            sigmaP = np.array(sigmaP)
        else:
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
        if not os.path.isfile(weight_file):
            np.save(weight_file, weights)
        if not os.path.isfile(indices_file):
            np.save(indices_file, indices_space)
    
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
        for i_subj in range(n_subj):
            timecourses = []
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
            data.append(pyrsa.data.Dataset(u_subj, obs_descriptors=desc))
        noise = pyrsa.data.prec_from_residuals(residuals)
        rdms = pyrsa.rdm.calc_rdm(data, method=rdm_type, descriptor='stim',
                                  cv_descriptor='repeat', noise=noise)
        rdms.save(rdm_file, file_type='hdf5')
    # get models
    models = get_models(
        model_type, stim_list,
        n_layer=12,
        smoothing=smoothing)
    # get true U RDMs
    # dat_true = []
    # for i_subj in range(U.shape[0]):
    #     dat_true.append(pyrsa.data.Dataset(Utrue[i_subj]))
    # rdms_true = pyrsa.rdm.calc_rdm(dat_true)
    # run inference & save it
    results = run_inference(models, rdms, method=rdm_comparison,
                            bootstrap=boot_type, n_cv=n_cv)
    results.save(full_path)
    print(full_path)
    

def fix_boot_cv(simulation_folder='boot_cv', ecoset_path='~/ecoset/val/'):
    """runs single flexible model simulations to allow parallelization
    """
    n_cvs = [1, 2, 4, 8, 16, 32]
    indices = np.random.permutation(len(n_cvs)* 100 * 10)
    for idx in indices:
        cv_idx = int(np.floor(idx / 1000))
        i_rep = int(np.floor((idx % 1000) / 10))
        i = int(idx % 10)
        boot_cv_sim(i=i, i_rep=i_rep, n_cv=n_cvs[cv_idx],
                    ecoset_path=ecoset_path,
                    simulation_folder=simulation_folder)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', type=str,
                        help='where is ecoset?', default=None)
    parser.add_argument('sim', help='simulation type',
                        choices=['comp', 'eco', 'flex', 'boot_cv',
                                 'summarize_eco', 'fix_eco'],
                        default='comp')
    parser.add_argument('index', type=int,
                        help='which simulation index to run')
    args = parser.parse_args()
    if args.sim == 'comp':
        run_comp(args.index)
    elif args.sim == 'eco':
        run_eco(args.index, ecoset_path=args.path)
    elif args.sim == 'summarize_eco':
        if args.path is None:
            summarize_eco()
        else:
            summarize_eco(args.path)
    elif args.sim == 'fix_eco':
        fix_eco(ecoset_path=args.path)
    elif args.sim == 'flex':
        if args.path is None:
            fix_flex()
        else:
            fix_flex(ecoset_path=args.path)
    elif args.sim == 'boot_cv':
        if args.path is None:
            fix_boot_cv()
        else:
            fix_boot_cv(ecoset_path=args.path)

