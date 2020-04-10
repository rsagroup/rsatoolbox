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
import scipy.special as special
import scipy.signal as signal
import scipy.linalg as linalg
import scipy.sparse
import scipy.stats as stats
from hrf import spm_hrf
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


def get_residuals(design, timecourse, beta):
    residuals = timecourse - np.matmul(design, beta)
    return residuals


def get_residuals_cross(designs, timecourses, betas):
    residuals = np.zeros_like(timecourses)
    for iCross in range(len(designs)):
        selected = np.ones(len(designs), 'bool')
        selected[iCross] = 0
        beta = np.mean(betas[selected], axis=0)
        residuals[iCross] = get_residuals(designs[iCross],
                                          timecourses[iCross],
                                          beta)
    return residuals


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
    print('\n getting true rdm')
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
        if get_cov_estimate:
            t = np.arange(0, 30, resolution)
            hrf = spm_hrf(t)
            hrf = np.array([hrf]).transpose()
            design = signal.convolve(design, hrf, mode='same')
            cov = calc_cov_estimate(betas[:, :len(stimuli)], sigma_p_est,
                                    sigmaP=np.einsum('ijk,ijl->kl',
                                                     residuals, residuals) \
                                            / residuals.shape[0]
                                            / residuals.shape[1],
                                    design=design, 
                                    sigmaKestimator=sigmaKestimator,
                                    sigmaRestimator=sigmaRestimator)
            covs.append(cov)
    return rdm_true, rdm_true_subj, rdm_samples, covs


def save_simulated_data_dnn(model=dnn.get_default_model(), layer=2, sd=3,
                            stimList=get_stimuli_96(), n_voxel=100, n_subj=10,
                            simulation_folder='sim', n_sim=1000, n_repeat=2,
                            duration=5, pause=1, endzeros=25,
                            use_cor_noise=True, resolution=2,
                            sigma_noise = 1, ar_coeff = .5):
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
        tim= []
        for i_subj in range(n_subj):
            (Utrue_subj,sigmaP_subj, indices_space_subj, weights_subj) = \
                dnn.get_sampled_representations(model, layer, [sd, sd],
                                                stimList, n_voxel)
            Utrue_subj = Utrue_subj / np.sqrt(np.sum(Utrue_subj ** 2)) \
                * np.sqrt(Utrue_subj.size)
            designs = []
            timecourses = []
            Usamps = []
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
            U.append(np.array(Usamps))
            des.append(np.array(designs))
            tim.append(np.array(timecourses))
            Utrue.append(Utrue_subj)
            sigmaP.append(sigmaP_subj)
            indices_space.append(indices_space_subj)
            weights.append(weights_subj)
        Utrue = np.array(Utrue)
        sigmaP = np.array(sigmaP)
        indices_space = np.array(indices_space)
        weights = np.array(weights)
        U = np.array(U)
        des = np.array(des)
        tim = np.array(tim)
        np.save(fname_base + 'Utrue%04d' % i, Utrue)
        np.save(fname_base + 'sigmaP%04d' % i, sigmaP)
        np.save(fname_base + 'indices_space%04d' % i, indices_space)
        np.save(fname_base + 'weights%04d' % i, weights)
        np.save(fname_base + 'U%04d' % i, U)

        
def analyse_saved_dnn(layer=2, sd=3, n_voxel=100,
                      n_subj=10, simulation_folder='sim', n_sim=100, n_repeat=2,
                      duration=5, pause=1, endzeros=25, use_cor_noise=True,
                      resolution=2, sigma_noise=1, ar_coeff=0.5,
                      modelType='fixed', model_rdm='averagetrue',
                      rdm_comparison='cosine', n_Layer=12, k_pattern=3,
                      k_rdm=3, rdm_type='crossnobis', n_stimuli=92):
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
        rdm_type, modelType, model_rdm, rdm_comparison, n_stimuli,
        k_pattern, k_rdm)
    if not os.path.isdir(res_path):
        os.mkdir(res_path)
    models = []
    pat_desc = {'stim':np.arange(n_stimuli)}
    for i_layer in range(n_Layer):
        if model_rdm == 'averagetrue':
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
        if modelType == 'fixed':
            models.append(pyrsa.model.ModelFixed('Layer%02d' % i_layer,
                pyrsa.rdm.RDMs(rdm, pattern_descriptors=pat_desc)))
    for i in tqdm.trange(n_sim, position=1):
        U = np.load(fname_base + 'U%04d.npy' % i)
        data = []
        desc = {'stim': np.tile(np.arange(n_stimuli), n_repeat),
                'repeat': np.repeat(np.arange(n_repeat), n_stimuli)}
        for i_subj in range(U.shape[0]):
            u_subj = U[i_subj, :, :n_stimuli, :].reshape(n_repeat * n_stimuli,
                                                         n_voxel)
            data.append(pyrsa.data.Dataset(u_subj, obs_descriptors=desc))
        rdms = pyrsa.rdm.calc_rdm(data, method=rdm_type, descriptor='stim',
                                  cv_descriptor='repeat')
        results = pyrsa.inference.bootstrap_crossval(models, rdms,
            pattern_descriptor='stim', rdm_descriptor='index',
            k_pattern=k_pattern, k_rdm=k_rdm, method=rdm_comparison)
        results.save(res_path + '/res%04d.hdf5' % (i))


def plot_saved_dnn(layer=2, sd=3, n_voxel=100, idx=0,
                   n_subj=10, simulation_folder='sim', n_repeat=2,
                   duration=5, pause=1, endzeros=25, use_cor_noise=True,
                   resolution=2, sigma_noise=1, ar_coeff=0.5,
                   modelType='fixed', model_rdm='averagetrue',
                   rdm_comparison='cosine', n_Layer=12, k_pattern=3, k_rdm=3,
                   rdm_type='crossnobis', n_stimuli=92, fname_base=None):
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
        rdm_type, modelType, model_rdm, rdm_comparison, n_stimuli,
        k_pattern, k_rdm)
    results = pyrsa.inference.load_results(res_path + '/res%04d.hdf5' % idx)
    pyrsa.vis.plot_model_comparison(results)


def get_fname_base(simulation_folder, layer, n_voxel, n_subj, n_repeat, sd,
                   duration, pause, endzeros, use_cor_noise, resolution,
                   sigma_noise, ar_coeff):
    """ generates the filename base from parameters """
    fname_base = simulation_folder + ('/layer%02d' % layer) \
        + ('/pars_%03d_%02d_%02d_%.2f/' % (n_voxel, n_subj, n_repeat, sd)) \
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


def calc_cov_estimate(us, sigma_p_est, sigmaP, design=None,
                      sigmaKestimator='eye', sigmaRestimator='eye'):
    # the design should already be convolved with the hrf!
    # this is only necessary for 'design' as sigmaK estimator
    u = np.mean(us,axis=0)
    n_repeat = us.shape[0]
    if sigmaKestimator == 'eye':
        sigmaK = np.eye(u.shape[0])
    elif sigmaKestimator == 'sample':
        sigmaK = pyrsa.get_sigmak(us)
    elif sigmaKestimator == 'design':
        sigmaK = np.linalg.inv(np.matmul(design.transpose(),design))
    else:
        raise ValueError('sigmaKestimator should be \'eye\' or \'sample\' ' \
                         'or \'design\'')
    if sigmaRestimator == 'eye':
        sigmaR = np.eye(u.shape[1])
    elif sigmaRestimator == 'sample':
        phalfinv = linalg.sqrtm(np.linalg.inv(sigma_p_est))
        sigmaR = np.matmul(np.matmul(phalfinv, sigmaP), phalfinv)
    else:
        raise ValueError('sigmaRestimator should be \'eye\' or \'sample\'') 
    #cov = scipy.sparse.csc_matrix(pyrsa.likelihood_cov(u,sigmaK,sigmaR,Nrepeat))
    return pyrsa.likelihood_cov(u, sigmaK, sigmaR, n_repeat)


# sampling distribution of one rdm
def sampling_one_rdm(U0=None, N=1000, sigmaP=None, shrinkage=0.4,
                     sigma=0.1, M=5, P=4):
    if U0 is None:
        U0 = pyrsa.generate_random_data(None, sigma=1, n_reps=1)
    if sigmaP is None:
        sigmaP = sigma * np.eye(U0.shape[-1])
    if len(U0.shape) == 3:
        U0 = np.mean(U0, axis=0)
        
    sigma_p_est, sigmaR = pyrsa.shrink_sigma_residual(sigmaP,
                                                      shrinkage=shrinkage)
    sigmaK = np.eye(U0.shape[0])
    rdm_theory = pyrsa.calc_rdm_mahalanobis(U0,sigma_p_est)
    cov_theory = pyrsa.likelihood_cov(U0,sigmaK,sigmaR,M)
    rdm_samples = np.zeros((N,rdm_theory.shape[0]))
    covs = np.zeros((N, cov_theory.shape[0], cov_theory.shape[1]))
    for i in range(N):
        Usamp = pyrsa.generate_random_data(U0, sigma=sigma, n_reps=M)
        rdm_samples[i],covs[i] = pyrsa.extract_from_u(Usamp, sigmaP,
                                                      shrinkage=shrinkage,
                                                      sigmaK=sigmaK)
    return rdm_theory,cov_theory,rdm_samples,covs




def explore_cov_formula(u=np.array([[0, 0, 0, 0, 0], [1, 2, 3, 0, 0],
                                    [4, 4, 4, 0, 0], [5, 5, 5, 5, 0]]),
                        N=100000, M=3, n_rep=100):
    ## playground
    P = u.shape[1]
    K = u.shape[0]
    C = pyrsa.get_cotrast_matrix(K)
    deltas = np.matmul(C, u)
    rdmtheory = np.sum(np.matmul(C,u) ** 2, axis=1) / P
    mapMat = np.eye(P)
    mapMat[1, 3] = 1
    sigmaRtheory = np.matmul(mapMat, mapMat.transpose())
    sigmaKtheory = np.eye(K)
    sigmaP = np.eye(P)
    Delta = np.matmul(np.matmul(C, u), np.matmul(C, u).transpose()) / P
    Xi = np.matmul(np.matmul(C,sigmaKtheory),C.transpose())
    
    n_ds = int(K*(K-1)/2)
    
    covtheory = np.zeros((n_ds, n_ds))
    for i in range(n_ds):
        for j in range(n_ds):
            covtheory[i, j] = (np.trace(np.matmul(sigmaRtheory,
                                                  sigmaRtheory))
                               / (P ** 2)
                               * (4 * Delta[i, j] * Xi[i, j] / M 
                                  + 2 * Xi[i, j] * Xi[i, j] / M / (M-1)))
    covMatrix = (np.sum(sigmaRtheory**2)/(P**2)*(4*Delta*Xi/M+2*Xi*Xi/M/(M-1)))
    
    ## My version 
    number = np.matmul(C,np.matmul(sigmaKtheory,C.transpose()))
    D = np.matmul(np.matmul(deltas,sigmaRtheory),deltas.transpose())
    cov_pred = 4/(P**2*M)*number*D + 2*(number**2)*np.sum(sigmaRtheory**2)/M/(M-1)/(P**2)
    
    covrdms = np.zeros((n_rep,cov_pred.shape[0],cov_pred.shape[0]))
    for iRep in tqdm.trange(n_rep):
        usamp = np.squeeze(np.matmul(mapMat,np.random.randn(M,N,4,5,1))) + u
        Ds = np.matmul(C,usamp) # difference vectors
        rdms = np.zeros((M,N,6))
        for im in range(M):
            idx = np.arange(M) !=im
            rdms[im] = np.einsum('ijk,ijk->ij',Ds[im],np.mean(Ds[idx],axis=0))/P
        rdms = np.mean(rdms,axis=0)
        covrdms[iRep] = np.cov(rdms.transpose())
    
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    for iRep in range(n_rep):
        plt.plot(covrdms[iRep][:],cov_pred[:],'k.')
    plt.plot([-5,30],[-5,30],'k--')
    plt.title('ours',fontsize=24)
    plt.xlabel('sampled/true covariance',fontsize = 15)
    plt.ylabel('formula covariance',fontsize = 15)
    plt.gca().set_aspect('equal', adjustable='box')
    
    plt.subplot(1,2,2)
    for iRep in range(n_rep):
        plt.plot(covrdms[iRep][:],covMatrix[:],'k.')
    plt.plot([-5,30],[-5,30],'k--')
    plt.title('Diedrichsen et. al.',fontsize=24)
    plt.xlabel('sampled/true covariance',fontsize = 15)
    plt.ylabel('formula covariance',fontsize = 15)
    plt.gca().set_aspect('equal', adjustable='box')
    
    plt.figure(figsize=(10,10))
    plt.plot(cov_pred[:],covMatrix[:],'k.')
    plt.plot([-5,30],[-5,30],'k--')
    plt.xlabel('ours',fontsize = 15)
    plt.ylabel('Diedrichsen et. al.',fontsize = 15)
    plt.gca().set_aspect('equal', adjustable='box')


def explore_cov_formula_random(N=100000, M=3, P=5, K=4, n_rep=100, mapMat=None):
    ## playground
    C = pyrsa.get_cotrast_matrix(K)
    if mapMat is None:
        mapMat = np.eye(P)
        mapMat[1,3] = 1
    
    sigmaRtheory = np.matmul(mapMat, mapMat.transpose())
    sigmaKtheory = np.eye(K)
    sigmaP = np.eye(P)

    n_ds = int(K * (K-1) / 2)

    covrdms = np.zeros((n_rep,n_ds,n_ds))
    covMatrix = np.zeros((n_rep,n_ds,n_ds))
    cov_pred = np.zeros((n_rep,n_ds,n_ds))
    for iRep in tqdm.trange(n_rep):
        u = np.random.randn(K,P)
        deltas = np.matmul(C,u)
        
        ## Diedrichsen
        Delta = np.matmul(np.matmul(C,u),np.matmul(C,u).transpose())/P
        Xi = np.matmul(np.matmul(C,sigmaKtheory),C.transpose())
        covMatrix[iRep] = (np.sum(sigmaRtheory ** 2) / (P ** 2) \
                           * (4 * Delta * Xi / M + 2 * Xi * Xi / M / (M - 1)))
        ## My version 
        number = np.matmul(C,np.matmul(sigmaKtheory, C.transpose()))
        D = np.matmul(np.matmul(deltas, sigmaRtheory), deltas.transpose())
        cov_pred[iRep] = 4 / (P ** 2 * M) * number * D \
            + 2 * (number ** 2) * np.sum(sigmaRtheory ** 2) / M / (M-1) / (P ** 2)
        usamp = np.squeeze(np.matmul(mapMat, np.random.randn(M, N, K, P, 1))) + u
        Ds = np.matmul(C,usamp) # difference vectors
        rdms = np.zeros((M, N, 6))
        for im in range(M):
            idx = np.arange(M) != im
            rdms[im] = np.einsum('ijk,ijk->ij', Ds[im],
                                 np.mean(Ds[idx], axis=0)) / P
        rdms = np.mean(rdms,axis=0)
        covrdms[iRep] = np.cov(rdms.transpose())
    
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    for iRep in range(n_rep):
        plt.plot(covrdms[iRep][:],cov_pred[iRep][:], 'k.')
    plt.plot([np.min(covrdms),np.max(covrdms)],
             [np.min(covrdms),np.max(covrdms)],
             'k--')
    plt.title('ours', fontsize=24)
    plt.xlabel('sampled/true covariance', fontsize=15)
    plt.ylabel('formula covariance', fontsize=15)
    plt.gca().set_aspect('equal', adjustable='box')
    
    plt.subplot(1,2,2)
    for iRep in range(n_rep):
        plt.plot(covrdms[iRep][:],covMatrix[iRep][:],'k.')
    plt.plot([np.min(covrdms), np.max(covrdms)],
             [np.min(covrdms), np.max(covrdms)],
             'k--')
    plt.title('Diedrichsen et. al.',fontsize=24)
    plt.xlabel('sampled/true covariance',fontsize = 15)
    plt.ylabel('formula covariance',fontsize = 15)
    plt.gca().set_aspect('equal', adjustable='box')
    
    plt.figure(figsize=(10,10))
    for iRep in range(n_rep):
        plt.plot(cov_pred[iRep][:],covMatrix[iRep][:],'k.')
    plt.plot([np.min(covrdms), np.max(covrdms)],
             [np.min(covrdms), np.max(covrdms)],
             'k--')
    plt.xlabel('ours',fontsize = 15)
    plt.ylabel('Diedrichsen et. al.',fontsize = 15)
    plt.gca().set_aspect('equal', adjustable='box')
    return covMatrix,cov_pred,covrdms
    
    
def test_cov_mult():
    # testing one equation
    N=100000
    A = np.random.randn(N, 2)
    B = np.random.randn(N, 2)
    C = A + np.random.randn(N, 2) + np.array([np.zeros(N), A[:, 0]]).transpose()
    D = B + np.random.randn(N, 2) + np.array([np.zeros(N), B[:, 0]]).transpose()
    return np.cov(np.sum(A * B, axis=1), np.sum(C * D, axis=1))
    

def plot_n_zeros(k=np.arange(4,50)):
    nzero = special.comb(k,2)*special.comb(k-2,2)
    nall = (k*(k-1)/2)**2
    plt.plot(k,nzero/nall)
    plt.xlabel('# of stimuli',FontSize=14)
    plt.ylabel('proportion of zeros in the whole matrix',FontSize=14)
    plt.ylim([0,1])

    
def pipeline_pooling_dnn(model=None, n_vox=10, fname='alexnet', n_subj=10,
                         n_repeat=5, cross_residuals=True, shrinkage=0.4,
                         layer=3, n_samp=3, n_stimuli=92):
    # runs the simulations to calculate a specific result for the pooling analysis
    # this produces 7 pooled rdms for every sample:
    # 0 : Mean
    # 1 : sigmaR = eye, sigmaK = eye
    # 2 : sigmaR = eye, sigmaK = design
    # 3 : sigmaR = eye, sigmaK = estimated
    # 4 : sigmaR = estimate, sigmaK = eye
    # 5 : sigmaR = estimate, sigmaK = design
    # 6 : sigmaR = estimate, sigmaK = estimated
    import nn_simulations as dnn
    if model is None:
        model = dnn.get_default_model()
    stimuli = get_stimuli_96()
    stimuli = stimuli[:n_stimuli]
    path = 'simulations/pooling_dnn'
    fname = path+'/'+fname
    if n_stimuli == 92:
        fnameTrue = fname+'True%d_%04d_%02dS_%02dR' % (
            layer, n_vox, n_subj, n_repeat)
        fnameSubj = fname+'Subj%d_%04d_%02dS_%02dR' % (
            layer, n_vox, n_subj, n_repeat)
        fnameSamples = fname+'Samp%d_%04d_%02dS_%02dR' % (
            layer, n_vox, n_subj, n_repeat)
    else:
        fnameTrue = fname+'True%d_%04d_%02d_%02dS_%02dR' % (
            layer, n_vox, n_stimuli, n_subj, n_repeat)
        fnameSubj = fname+'Subj%d_%04d_%02d_%02dS_%02dR' % (
            layer, n_vox, n_stimuli, n_subj, n_repeat)
        fnameSamples = fname+'Samp%d_%04d_%02d_%02dS_%02dR' % (
            layer, n_vox, n_stimuli, n_subj, n_repeat)
    repeats = 1
    duration = 5
    pause = 1
    endzeros = 20
    sd = np.array([5, 5])
    use_cor_noise = True
    sigma_noise = None
    resolution = None
    ar_coeff = None
    rdms_true_subj = []
    rdms_samples = []
    rdms_covs = []
    for iSamp in tqdm.trange(n_samp, position=0):
        rdm_true = dnn.get_true_rdm(model,layer,stimuli)
        rdm_true_subj = np.zeros((n_subj,)+rdm_true.shape)
        rdm_samples = []
        covs11 = []
        covs12 = []
        covs13 = []
        covs21 = []
        covs22 = []
        covs23 = []
        total_dur = len(stimuli)*repeats*(duration+pause)+endzeros
        #print('\n starting simulations')
        for i_subj in tqdm.trange(n_subj,position=1):
            Usubj, sigmaP, indices_space, weights = \
                dnn.get_sampled_representations(model, layer, sd, stimuli,
                                                n_vox)
            rdm_true_subj[i_subj] = pyrsa.calc_rdm_mahalanobis(Usubj,sigmaP)
            # replaced with proper sampling of a timecourse
            #Usamp = dnn.get_random_sample(Usubj,sigmaP,sigmaNoise,N) 
            rdm_samples_subj = np.zeros((len(stimuli), len(stimuli)))
            betas = np.zeros((n_repeat, len(stimuli) + 1, n_vox))
            timecourses = np.zeros((n_repeat,total_dur, n_vox))
            designs = np.zeros((n_repeat, total_dur, len(stimuli) + 1))
            for i_repeat in range(n_repeat):
                design = dnn.generate_design_random(len(stimuli), repeats=1,
                                                    duration=duration,
                                                    pause=pause,
                                                    endzeros=endzeros)
                if use_cor_noise:
                    timecourse = dnn.generate_timecourse(design, Usubj,
                                                         sigma_noise,
                                                         resolution=resolution,
                                                         ar_coeff=ar_coeff,
                                                         sigmaP=sigmaP)
                else:
                    timecourse = dnn.generate_timecourse(design, Usubj,
                                                         sigma_noise,
                                                         resolution=resolution,
                                                         ar_coeff=ar_coeff,
                                                         sigmaP=None)
                betas[i_repeat] = estimate_betas(design,timecourse)
                timecourses[i_repeat] = timecourse
                designs[i_repeat] = design
            if cross_residuals:
                residuals = get_residuals_cross(designs,timecourses,betas)
            else:
                residuals = get_residuals(designs,timecourses,betas)
            if shrinkage ==0:
                sigma_p_est = np.einsum('ijk,ijl->kl', residuals, residuals) \
                    / residuals.shape[0] / residuals.shape[1]
            elif shrinkage <=1 :
                sigma_p_est = np.einsum('ijk,ijl->kl', residuals, residuals) \
                    / residuals.shape[0] / residuals.shape[1]
                sigma_p_est = shrinkage * np.diag(np.diag(sigma_p_est)) \
                    + (1-shrinkage)* sigma_p_est
            elif shrinkage ==np.inf:
                sigma_p_est = np.eye(n_vox)
            rdm_samples_subj = pyrsa.calc_rdm_crossnobis(
                betas[:, :len(stimuli)], sigma_p_est)
            rdm_samples.append(rdm_samples_subj) 
            t = np.arange(0,30,resolution)
            hrf = spm_hrf(t)
            hrf = np.array([hrf]).transpose()
            design = signal.convolve(design,hrf,mode = 'full')[:design.shape[0]]
            design = design[:,:len(stimuli)]
            covs11.append(calc_cov_estimate(betas[:,:len(stimuli)],
                sigma_p_est,
                sigmaP=np.einsum('ijk,ijl->kl', residuals, residuals)
                    / residuals.shape[0] / residuals.shape[1],
                design=design,
                sigmaKestimator='eye',
                sigmaRestimator='eye'))
            covs12.append(calc_cov_estimate(betas[:, :len(stimuli)],
                                            sigma_p_est,
                                            sigmaP=np.einsum('ijk,ijl->kl',
                                                             residuals,
                                                             residuals)
                                                    / residuals.shape[0]
                                                    / residuals.shape[1],
                                            design=design,
                                            sigmaKestimator='design',
                                            sigmaRestimator='eye'))
            covs13.append(calc_cov_estimate(betas[:, :len(stimuli)],
                                            sigma_p_est,
                                            sigmaP=np.einsum('ijk,ijl->kl',
                                                             residuals,
                                                             residuals)
                                                    / residuals.shape[0]
                                                    / residuals.shape[1],
                                            design=design,
                                            sigmaKestimator='sample',
                                            sigmaRestimator='eye'))
            covs21.append(calc_cov_estimate(betas[:,:len(stimuli)],
                                            sigma_p_est,
                                            sigmaP=np.einsum('ijk,ijl->kl',
                                                             residuals,
                                                             residuals)
                                                   / residuals.shape[0]
                                                   / residuals.shape[1],
                                            design=design,
                                            sigmaKestimator='eye',
                                            sigmaRestimator='sample'))
            covs22.append(calc_cov_estimate(betas[:,:len(stimuli)],
                                            sigma_p_est,
                                            sigmaP=np.einsum('ijk,ijl->kl',
                                                             residuals,
                                                             residuals)
                                                   / residuals.shape[0]
                                                   / residuals.shape[1],
                                            design=design,
                                            sigmaKestimator = 'design',
                                            sigmaRestimator = 'sample'))
            covs23.append(calc_cov_estimate(betas[:, :len(stimuli)],
                                            sigma_p_est,
                                            sigmaP=np.einsum('ijk,ijl->kl',
                                                             residuals,
                                                             residuals)
                                                   / residuals.shape[0]
                                                   / residuals.shape[1],
                                                   design=design,
                                            sigmaKestimator='sample',
                                            sigmaRestimator='sample'))
        #print('\n starting pooling')
        rdm_samples = np.array(rdm_samples)
        rdms_true_subj.append(rdm_true_subj)
        rdm_pool = []
        cov_pool = []
        pool = pool_rdms(rdm_samples)
        rdm_pool.append(pyrsa.get_rdm_vector(pool))
        pool = pool_rdms(rdm_samples, covs11)
        rdm_pool.append(pool[0])
        cov_pool.append(pool[1])
        pool = pyrsa.pool_rdms(rdm_samples, covs12)
        rdm_pool.append(pool[0])
        cov_pool.append(pool[1])
        pool = pool_rdms(rdm_samples, covs13)
        rdm_pool.append(pool[0])
        cov_pool.append(pool[1])
        pool = pool_rdms(rdm_samples, covs21)
        rdm_pool.append(pool[0])
        cov_pool.append(pool[1])
        pool = pool_rdms(rdm_samples, covs22)
        rdm_pool.append(pool[0])
        cov_pool.append(pool[1])
        pool = pool_rdms(rdm_samples, covs23)
        rdm_pool.append(pool[0])
        cov_pool.append(pool[1])
        rdms_samples.append(np.array(rdm_pool))
        rdms_covs.append(cov_pool)
    rdms_samples = np.array(rdms_samples)
    rdms_true_subj = np.array(rdms_true_subj)
    np.save(fnameTrue, rdm_true)
    np.save(fnameSubj, rdms_true_subj)
    np.save(fnameSamples, rdms_samples)


def plot_dnn_pooling(fname='alexnet', n_stimuli=40, layer=3, n_subj=10,
                     n_repeat=5):
    path = 'simulations/pooling_dnn'
    fname = path+'/'+fname
    if n_stimuli == 92:
        fnameTrue = fname + 'True%d_%04d_%02dS_%02dR.npy' % (
            layer, 10, n_subj, n_repeat)
    else:
        fnameTrue = fname + 'True%d_%04d_%02d_%02dS_%02dR.npy' % (
            layer, 10, n_stimuli, n_subj, n_repeat)
    truerdm = np.load(fnameTrue)
    error = np.zeros((5, 7))
    corr = np.zeros((5, 7))
    errorSubj = np.zeros((5))
    corrSubj = np.zeros((5))
    k = 0
    for n_vox in (10, 25, 75, 200, 500):
        if n_stimuli == 92:
            fnameSubj = fname + 'Subj%d_%04d_%02dS_%02dR.npy' % (
                layer, n_vox, n_subj, n_repeat)
            fnameSamples = fname + 'Samp%d_%04d_%02dS_%02dR.npy' % (
                layer, n_vox, n_subj, n_repeat)
        else:
            fnameSubj = fname + 'Subj%d_%04d_%02d_%02dS_%02dR.npy' % (
                layer, n_vox, n_stimuli, n_subj, n_repeat)
            fnameSamples = fname+'Samp%d_%04d_%02d_%02dS_%02dR.npy' % (
                layer, n_vox, n_stimuli, n_subj, n_repeat)
        trueSubj = np.load(fnameSubj)
        n_samp = trueSubj.shape[0]
        trueSubj = trueSubj.reshape(n_subj * n_samp, n_stimuli, n_stimuli)
        samples = np.load(fnameSamples)
        errorSubj[k] = np.mean(pyrsa.error_rdm(trueSubj, truerdm))
        corrSubj[k] = np.mean(pyrsa.corr_rdm(trueSubj, truerdm))
        for i_pool in range(7):
            error[k, i_pool] = np.mean(pyrsa.error_rdm(samples[:, i_pool],
                                                      truerdm))
            corr[k, i_pool] = np.mean(pyrsa.corr_rdm(samples[:, i_pool],
                                                    truerdm))
        k = k+1
    plt.figure()
    plt.plot(corr)
    plt.xticks(ticks=[0, 1, 2, 3, 4], labels=['10', '25', '75', '200', '500'])
    plt.xlabel('Number of Voxels')
    plt.ylabel('rdm correlation')
    plt.legend(['simple Mean', 'I,I', 'I,design', 'I,sample', 'sample,I',
                'sample,design', 'sample,sample'], frameon=False)
    plt.figure()
    plt.plot(error)
    plt.xticks(ticks=[0, 1, 2, 3, 4], labels=['10', '25', '75', '200', '500'])
    plt.ylim(top=1, bottom=0)
    #plt.yscale('log')
    plt.xlabel('Number of Voxels')
    plt.ylabel('Normalized RMSE')
    plt.legend(['simple Mean', 'I,I', 'I,design', 'I,sample', 'sample,I',
                'sample,design', 'sample,sample'], frameon=False)


def get_rdm_vector(rdm):
    if len(rdm.shape) == 1: # assume it is one vector
        return rdm
    elif len(rdm.shape) == 2:
        if rdm.shape[0] == rdm.shape[1]: # it is a single RDM
            idx = np.triu_indices_from(rdm,1)
            return rdm[idx]
        # assume it already is a set of vectors
        return rdm
    elif len(rdm.shape)==3 and rdm.shape[1] == rdm.shape[2]: # it is a set of RDMs
        idx = np.triu_indices_from(rdm[0],1)
        return rdm[:, idx[0], idx[1]]
    else:
        raise(ValueError('get_rdm_vector received unrecognized shape'))


def get_rdm_matrix(rdm):
    if len(rdm.shape)==1:
        N = int(np.sqrt(2*rdm.shape[0]+0.25)+0.5)
        rdm_out = np.zeros((N,N))
        rdm_out[np.triu_indices_from(rdm_out,1)]=rdm
        #RDM[np.tril_indices_from(RDM,-1)]=rdm-> wrong indices!
        rdm_out = rdm_out+rdm_out.T
    elif len(rdm.shape)==2:
        if (rdm.shape[0] != rdm.shape[1]) or not np.allclose(rdm,rdm.transpose()):
            N = int(np.sqrt(2*rdm.shape[1]+0.25)+0.5)
            rdm_out = np.zeros((rdm.shape[0],N,N))
            idx = np.triu_indices_from(rdm_out[0],1)
            rdm_out[:,idx[0],idx[1]] = rdm
            rdm_out = rdm_out + rdm_out.transpose(0,2,1)
        else:
            rdm_out = rdm
    elif len(rdm.shape) == 3:
        if rdm.shape[1]==rdm.shape[2]:
            rdm_out = rdm
    else:
        raise(ValueError('get_rdm_matrix received unrecognized shape'))
    return rdm_out


def multiply_gaussians(mus, covs):
    mu = mus[0]
    if (type(covs[0]) is np.ndarray):
        precision = covs[0]
    else:
        precision = covs[0].toarray()   
    for i in range(1,len(mus)):
        if (type(covs[i]) is np.ndarray):
            prec_i = np.linalg.inv(covs[i])
        else:
            prec_i = np.linalg.inv(covs[i].toarray())
        prec_new = precision + prec_i
        mu = np.linalg.solve(prec_new,np.matmul(precision,mu)) \
            + np.linalg.solve(prec_new,np.matmul(prec_i,mus[i]))
        precision = prec_new
    cov = np.linalg.inv(precision)
    return mu, cov


def multiply_gaussians_normalize(mus, covs):
    mu = mus[0]
    precision = np.linalg.inv(covs[0])
    z = 1 / np.sqrt(np.linalg.det(covs[0])) \
        * np.exp(-0.5 * np.matmul(mu, np.matmul(precision, mu)))
    for i in range(1,len(mus)):
        prec_i = np.linalg.inv(covs[i])
        prec_new = precision + prec_i
        mu = np.linalg.solve(prec_new, np.matmul(precision, mu)) \
            + np.linalg.solve(prec_new, np.matmul(prec_i, mus[i]))
        precision = prec_new
        z = z / np.sqrt(np.linalg.det(covs[0])) \
            * np.exp(-0.5 * np.matmul(mus[i], np.matmul(prec_i, mus[i])))
    cov = np.linalg.inv(precision)
    z = z * np.sqrt(np.linalg.det(covs[0]))\
        * np.exp(0.5 * np.matmul(mu, np.matmul(precision, mu)))
    z = (2 * np.pi) ** (-len(mu) / 2) * z
    return mu, cov, z


def pool_rdms(rdms, covs=None):
    # will automatically return cov if you pass a covariance
    if len(rdms.shape) == 3:
        means = np.mean(np.mean(rdms, axis=1, keepdims=True),
                        axis=2, keepdims=True)
    else:
        means = np.mean(rdms, axis=1, keepdims=True)
    rdms = rdms / means
    if covs is None:
        rdm = np.mean(rdms, axis=0)
    else:
        for i in range(rdms.shape[0]):
            covs[i] = covs[i] / means[i] / means[i]
        rdm = pool_rdms_hard(rdms, covs)
    return rdm


def pool_rdms_hard(rdms, covs):
    matrixform = len(rdms.shape) == 3
    if matrixform:
        rdms = get_rdm_vector(rdms)
    rdm, cov = multiply_gaussians(rdms, covs)
    if matrixform:
        rdms = get_rdm_matrix(rdms)
    return rdm, cov
