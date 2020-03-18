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
    for iStim in range(92):
        im = PIL.Image.open('96Stimuli/stimulus%d.tif' % (iStim+1))
        stimuli.append(im)
    return stimuli


def get_stimuli_96():
    import PIL
    stimuli = []
    for iStim in range(96):
        im = PIL.Image.open('96Stimuli/stimulus%d.tif' % (iStim+1))
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

# sampling from a neural network layer
def sampling_DNN(stimuli, Nsubj=3, Nvox=100, Nrepeat=5, shrinkage=0,
                 model=None, layer=3, sd=np.array([5, 5]),
                 sigma_noise=None, resolution=None, ar_coeff=None,
                 repeats=1, duration=5, pause=1, endzeros=20,
                 cross_residuals=True, sigmaKestimator='eye',
                 sigmaRestimator='eye', use_cor_noise=True,
                 get_cov_estimate=True):
    # shrinkage governs how strongly sigmaP is shrunk, 0 is raw estimate,
    # 1 is only diagonal, np.inf is eye
    # Noise estimation choices:
    #     generating noise:
    #         - on representation-> correlated noise over voxels
    if model is None:
        model = dnn.get_default_model()
    print('\n getting true RDM')
    RDM_true = dnn.get_true_RDM(model, layer, stimuli)
    RDM_true_subj = np.zeros((Nsubj, ) + RDM_true.shape)
    RDM_samples = []
    covs = []
    total_dur = len(stimuli) * repeats * (duration + pause) + endzeros
    print('\n starting simulations')
    for iSubj in tqdm.trange(Nsubj):
        Usubj, sigmaP, indices_space, weights = \
            dnn.get_sampled_representations(
                model, layer, sd, stimuli, Nvox)
        RDM_true_subj[iSubj] = pyrsa.calc_RDM_mahalanobis(Usubj, sigmaP)
        # replaced with proper sampling of a timecourse
        #Usamp = dnn.get_random_sample(Usubj,sigmaP,sigmaNoise,N)
        rdm_samples_subj = np.zeros((len(stimuli), len(stimuli)))
        betas = np.zeros((Nrepeat, len(stimuli) + 1, Nvox))
        timecourses = np.zeros((Nrepeat, total_dur, Nvox))
        designs = np.zeros((Nrepeat, total_dur, len(stimuli) + 1))
        for iSamp in range(Nrepeat):
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
            sigma_p_est = np.eye(Nvox)
        rdm_samples_subj = pyrsa.calc_RDM_crossnobis(betas[:, :len(stimuli)],
                                                     sigma_p_est)
        RDM_samples.append(rdm_samples_subj)
        if calc_cov_estimate:
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
    RDM_samples = np.array(RDM_samples)
    return RDM_true, RDM_true_subj, RDM_samples, covs



def save_simulated_data_dnn(model=dnn.get_default_model(), layer=2, sd=3,
                            stimList=get_stimuli_96(), Nvoxel=100, Nsubj=10,
                            simulation_folder='test', Nsim=1000, Nrepeat=2,
                            duration=5, pause=1, endzeros=25,
                            use_cor_noise=True, resolution=2,
                            sigma_noise = 1, ar_coeff = .5):
    fname_base = simulation_folder + ('/layer%02d' % layer) \
        + ('/pars_%03d_%02d_%02d_%.2f/' % (Nvoxel,Nsubj,Nrepeat,sd)) \
        + ('fmri_%02d_%02d_%03d_%s_%d_%.2f_%.2f/' % (
            duration, pause, endzeros, use_cor_noise, resolution,
            sigma_noise, ar_coeff))
    if not os.path.isdir(fname_base):
        os.makedirs(fname_base)
    for i in tqdm.trange(Nsim):
        Utrue = []
        sigmaP = []
        indices_space = []
        weights = []
        U = []
        des = []
        tim= []
        for iSubj in range(Nsubj):
            (Utrue_subj,sigmaP_subj, indices_space_subj, weights_subj) = \
                dnn.get_sampled_representations(model, layer, [sd, sd],
                                                stimList, Nvoxel)
            Utrue_subj = Utrue_subj / np.mean(Utrue_subj)
            designs = []
            timecourses = []
            Usamps = []
            for iSamp in range(Nrepeat):
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
        #np.save(fname_base+'design%04d'%i,des)
        #np.save(fname_base+'timecourse%04d'%i,tim)
        
def analyse_saved_dnn(layer=2, sd=3, stimList=get_stimuli_96(), Nvoxel=100,
                      Nsubj=10, simulation_folder='test', Nsim=100, Nrepeat=2,
                      duration=5, pause=1, endzeros=25, use_cor_noise=True,
                      resolution=2, sigma_noise=1, ar_coeff=0.5,
                      modelType='fixed', model_RDM='average_true',
                      RDM_comparison='cosine', NLayer=7, nFold=5,
                      RDM_type='crossnobis', Nstimuli=92):
    fname_base = simulation_folder + ('/layer%02d' % layer) \
        + ('/pars_%03d_%02d_%02d_%.2f/' % (Nvoxel, Nsubj, Nrepeat, sd)) \
        + ('fmri_%02d_%02d_%03d_%s_%d_%.2f_%.2f/' % (duration, pause,
            endzeros, use_cor_noise, resolution, sigma_noise, ar_coeff))
    assert os.path.isdir(fname_base), 'simulated data not found!'
    models = []
    for iLayer in range(NLayer):
        if model_RDM == 'average_true':
            fname_baseL = simulation_folder + ('/layer%02d' % (iLayer+1)) \
                + ('/pars_%03d_%02d_%02d_%.2f/' % (Nvoxel, Nsubj, 
                                                   Nrepeat, sd)) \
                + ('fmri_%02d_%02d_%03d_%s_%d_%.2f_%.2f/' % (
                    duration, pause, endzeros, use_cor_noise, resolution,
                    sigma_noise, ar_coeff))
            RDMtrue_average = 0
            for i in range(Nsim):
                Utrue = np.load(fname_baseL + 'Utrue%04d.npy' % i)
                RDMtrue = pyrsa.calc_RDM(Utrue[:, :Nstimuli,:],
                                         method='euclid')
                RDMtrue = RDMtrue / np.mean(RDMtrue)
                RDMtrue_average = RDMtrue_average + np.mean(RDMtrue,0)
            RDM = RDMtrue_average/Nsim
        if modelType == 'fixed':
            models.append(pyrsa.model_fix(RDM))
    scores = []
    noise_ceilings = []
    for i in tqdm.trange(Nsim):
        U = np.load(fname_base + 'U%04d.npy' % i)
        RDMs = []
        for iSubj in range(U.shape[0]):
            RDMs.append(pyrsa.calc_RDM(U[iSubj,:,:Nstimuli,:], method=RDM_type))
        RDMs = np.array(RDMs)
        score = np.array([pyrsa.crossvalidate(m, RDMs, method=RDM_comparison,
                                              nFold=nFold)
                          for m in models])
        [noise_min,noise_max] = pyrsa.noise_ceiling(RDMs,
                                                    method=RDM_comparison,
                                                    nFold=nFold)
        np.save(fname_base + 'RDMs_%s_%04d.npy' % (RDM_type, i), RDMs)
        scores.append(score)
        noise_ceilings.append([noise_min,noise_max])
    scores = np.array(scores)
    noise_ceilings = np.array(noise_ceilings)
    np.save(fname_base + 'scores_%s_%s_%s_%s_%d_%d.npy' % (
        RDM_type, modelType, model_RDM, RDM_comparison, Nstimuli, nFold),
        scores)
    np.save(fname_base + 'noisec_%s_%s_%s_%s_%d_%d.npy' % (
        RDM_type, modelType, model_RDM, RDM_comparison, Nstimuli, nFold),
        noise_ceilings)


def plot_saved_dnn(layer=2, sd=3, stimList=get_stimuli_96(), Nvoxel=100,
                   Nsubj=10, simulation_folder='test', Nsim=100, Nrepeat=2,
                   duration=5, pause=1, endzeros=25, use_cor_noise = True,
                   resolution=2, sigma_noise=2, ar_coeff=0.5,
                   modelType='fixed', model_RDM='average_true',
                   RDM_comparison='cosine', NLayer=12, nFold=5,
                   RDM_type='crossnobis', Nstimuli=96, fname_base=None):
    if fname_base is None:
        fname_base = simulation_folder + ('/layer%02d' % layer) \
            + ('/pars_%03d_%02d_%02d_%.2f/' % (Nvoxel, Nsubj, Nrepeat, sd)) \
            + ('fmri_%02d_%02d_%03d_%s_%d_%.2f_%.2f/' % (
                duration, pause, endzeros, use_cor_noise, resolution,
                sigma_noise, ar_coeff))
    assert os.path.isdir(fname_base), 'simulated data not found!'
    scores = np.load(fname_base + 'scores_%s_%s_%s_%s_%d_%d.npy' % (
        RDM_type, modelType, model_RDM, RDM_comparison, Nstimuli, nFold))
    noise_ceilings = np.load(fname_base + 'noisec_%s_%s_%s_%s_%d_%d.npy' % (
        RDM_type,modelType,model_RDM,RDM_comparison,Nstimuli,nFold))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.tick_params(labelsize=12)
    for iSim in range(Nsim):
        ax.fill_between(np.array([0.5, NLayer + 0.5]), noise_ceilings[iSim, 0],
                        noise_ceilings[iSim, 1], facecolor='blue',
                        alpha=1 / Nsim)
    #ax.plot(np.array([0.5,NLayer+0.5]),np.repeat(noise_ceilings[:,0],2).reshape([Nsim,2]).T,'k',alpha=.1)
    #ax.plot(np.array([0.5,NLayer+0.5]),np.repeat(noise_ceilings[:,1],2).reshape([Nsim,2]).T,'k',alpha=.1)
    for iFold in range(nFold):
        ax.plot(np.arange(NLayer) + 1 - nFold / 20 + 0.1 * iFold,
                scores[:, :NLayer, iFold].T, 'k.')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel('Layer', fontsize=18)
    ax.set_title('Layer %d' % layer, fontsize=28)
    if RDM_comparison=='cosine':
        plt.ylim([0, 1])
        ax.set_ylabel('Cosine Distance', fontsize=18)
    elif RDM_comparison=='eudlid':
        ax.set_ylabel('Euclidean Distance', fontsize=18)
    elif RDM_comparison=='kendall-tau':
        ax.set_ylabel('Kendall Tau', fontsize=18)
    elif RDM_comparison=='pearson':
        ax.set_ylabel('Pearson Correlation', fontsize=18)
    elif RDM_comparison=='spearman':
        ax.set_ylabel('Spearman Rho', fontsize=18)
        
def plot_saved_dnn_average(layer=2, sd=3, stimList=get_stimuli_96(),
                           Nvoxel=100, Nsubj=10, simulation_folder='test',
                           Nsim=100, Nrepeat=2, duration=5, pause=1,
                           endzeros=25, use_cor_noise=True, resolution = 2,
                           sigma_noise=2, ar_coeff=.5, modelType = 'fixed',
                           model_RDM = 'average_true', Nstimuli=96,
                           RDM_comparison = 'cosine', NLayer = 12, nFold=5,
                           RDM_type='crossnobis', fname_base=None):
    if fname_base is None:
        fname_base = simulation_folder + ('/layer%02d' % layer) \
            + ('/pars_%03d_%02d_%02d_%.2f/' % (Nvoxel, Nsubj, Nrepeat, sd)) \
            + ('fmri_%02d_%02d_%03d_%s_%d_%.2f_%.2f/' % (
                duration, pause, endzeros, use_cor_noise, resolution,
                sigma_noise, ar_coeff))
    assert os.path.isdir(fname_base), 'simulated data not found!'
    scores = np.load(fname_base + 'scores_%s_%s_%s_%s_%d_%d.npy' % (
        RDM_type, modelType, model_RDM, RDM_comparison, Nstimuli, nFold))
    noise_ceilings = np.load(fname_base + 'noisec_%s_%s_%s_%s_%d_%d.npy' % (
        RDM_type, modelType, model_RDM, RDM_comparison, Nstimuli, nFold))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.tick_params(labelsize=12)
    for iSim in range(Nsim):
        ax.fill_between(np.array([0.5, NLayer + 0.5]), noise_ceilings[iSim, 0],
                        noise_ceilings[iSim, 1], facecolor='blue',
                        alpha=1 / Nsim)
    #ax.plot(np.array([0.5,NLayer+0.5]),np.repeat(noise_ceilings[:,0],2).reshape([Nsim,2]).T,'k',alpha=.1)
    #ax.plot(np.array([0.5,NLayer+0.5]),np.repeat(noise_ceilings[:,1],2).reshape([Nsim,2]).T,'k',alpha=.1)
    ax.plot(np.arange(NLayer) + 1 - nFold / 20, np.mean(scores[:, :NLayer, :],
                                                        axis=2).T,
            'k.')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel('Layer', fontsize=18)
    ax.set_title('Layer %d' % layer, fontsize=28)
    if RDM_comparison=='cosine':
        plt.ylim([0,1])
        ax.set_ylabel('Cosine Distance', fontsize=18)
    elif RDM_comparison=='eudlid':
        ax.set_ylabel('Euclidean Distance', fontsize=18)
    elif RDM_comparison=='kendall-tau':
        ax.set_ylabel('Kendall Tau', fontsize=18)
    elif RDM_comparison=='pearson':
        ax.set_ylabel('Pearson Correlation', fontsize=18)
    elif RDM_comparison=='spearman':
        ax.set_ylabel('Spearman Rho', fontsize=18)

def calc_cov_estimate(us,sigma_p_est, sigmaP, design=None,
                      sigmaKestimator='eye', sigmaRestimator='eye'):
    # the design should already be convolved with the hrf!
    # this is only necessary for 'design' as sigmaK estimator
    u = np.mean(us,axis=0)
    Nrepeat = us.shape[0]
    if sigmaKestimator == 'eye':
        sigmaK = np.eye(u.shape[0])
    elif sigmaKestimator == 'sample':
        sigmaK = pyrsa.get_sigmak(us)
    elif sigmaKestimator == 'design':
        sigmaK = np.linalg.inv(np.matmul(design.transpose(),design))
    else:
        raise ValueError('sigmaKestimator should be \'eye\' or \'sample\' or \'design\'')   
       
    if sigmaRestimator == 'eye':
        sigmaR = np.eye(u.shape[1])
    elif sigmaRestimator == 'sample':
        phalfinv = linalg.sqrtm(np.linalg.inv(sigma_p_est))
        sigmaR = np.matmul(np.matmul(phalfinv, sigmaP), phalfinv)
    else:
        raise ValueError('sigmaRestimator should be \'eye\' or \'sample\'') 
    #cov = scipy.sparse.csc_matrix(pyrsa.likelihood_cov(u,sigmaK,sigmaR,Nrepeat))
    return pyrsa.likelihood_cov(u,sigmaK,sigmaR,Nrepeat)


# sampling distribution of one RDM
def sampling_one_RDM(U0 = None, N = 1000, sigmaP = None, shrinkage=0.4, sigma=0.1,M=5,P=4):
    if U0 is None:
        U0 = pyrsa.generate_random_data(None,sigma=1,Nreps=1)
    if sigmaP is None:
        sigmaP = sigma*np.eye(U0.shape[-1])
    if len(U0.shape)==3:
        U0 = np.mean(U0,axis=0)
        
    sigma_p_est,sigmaR = pyrsa.shrink_sigma_residual(sigmaP,shrinkage=shrinkage)
    sigmaK = np.eye(U0.shape[0])
    RDM_theory = pyrsa.calc_RDM_mahalanobis(U0,sigma_p_est)
    #RDM_theory,cov_theory = pyrsa.extract_from_u(U0,sigmaP,shrinkage=shrinkage)
    #print(sigmaR)
    cov_theory = pyrsa.likelihood_cov(U0,sigmaK,sigmaR,M)
    RDM_samples = np.zeros((N,RDM_theory.shape[0]))
    covs = np.zeros((N,cov_theory.shape[0],cov_theory.shape[1]))
    for i in range(N):
        Usamp = pyrsa.generate_random_data(U0,sigma=sigma,Nreps=M)
        RDM_samples[i],covs[i] = pyrsa.extract_from_u(Usamp,sigmaP,shrinkage=shrinkage,sigmaK=sigmaK)
    return RDM_theory,cov_theory,RDM_samples,covs




def explore_cov_formula(u=np.array([[0, 0, 0, 0, 0], [1, 2, 3, 0, 0],
                                    [4, 4, 4, 0, 0], [5, 5, 5, 5, 0]]),
                        N=100000, M=3, Nrep=100):
    ## playground
    P = u.shape[1]
    K = u.shape[0]
    C = pyrsa.get_cotrast_matrix(K)
    deltas = np.matmul(C, u)
    RDMtheory = np.sum(np.matmul(C,u) ** 2, axis=1) / P
    mapMat = np.eye(P)
    mapMat[1, 3] = 1
    sigmaRtheory = np.matmul(mapMat, mapMat.transpose())
    sigmaKtheory = np.eye(K)
    sigmaP = np.eye(P)
    Delta = np.matmul(np.matmul(C, u), np.matmul(C, u).transpose()) / P
    Xi = np.matmul(np.matmul(C,sigmaKtheory),C.transpose())
    
    Nds = int(K*(K-1)/2)
    
    covtheory = np.zeros((Nds,Nds))
    for i in range(Nds):
        for j in range(Nds):
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
    
    covRDMs = np.zeros((Nrep,cov_pred.shape[0],cov_pred.shape[0]))
    for iRep in tqdm.trange(Nrep):
        usamp = np.squeeze(np.matmul(mapMat,np.random.randn(M,N,4,5,1))) + u
        Ds = np.matmul(C,usamp) # difference vectors
        RDMs = np.zeros((M,N,6))
        for im in range(M):
            idx = np.arange(M) !=im
            RDMs[im] = np.einsum('ijk,ijk->ij',Ds[im],np.mean(Ds[idx],axis=0))/P
        RDMs = np.mean(RDMs,axis=0)
        covRDMs[iRep] = np.cov(RDMs.transpose())
    
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    for iRep in range(Nrep):
        plt.plot(covRDMs[iRep][:],cov_pred[:],'k.')
    plt.plot([-5,30],[-5,30],'k--')
    plt.title('ours',fontsize=24)
    plt.xlabel('sampled/true covariance',fontsize = 15)
    plt.ylabel('formula covariance',fontsize = 15)
    plt.gca().set_aspect('equal', adjustable='box')
    
    plt.subplot(1,2,2)
    for iRep in range(Nrep):
        plt.plot(covRDMs[iRep][:],covMatrix[:],'k.')
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
    
def explore_cov_formula_random(N=100000, M=3, P=5, K=4, Nrep=100, mapMat=None):
    ## playground
    C = pyrsa.get_cotrast_matrix(K)
    if mapMat is None:
        mapMat = np.eye(P)
        mapMat[1,3] = 1
    
    sigmaRtheory = np.matmul(mapMat, mapMat.transpose())
    sigmaKtheory = np.eye(K)
    sigmaP = np.eye(P)

    Nds = int(K * (K-1) / 2)

    covRDMs = np.zeros((Nrep,Nds,Nds))
    covMatrix = np.zeros((Nrep,Nds,Nds))
    cov_pred = np.zeros((Nrep,Nds,Nds))
    for iRep in tqdm.trange(Nrep):
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
        RDMs = np.zeros((M, N, 6))
        for im in range(M):
            idx = np.arange(M) != im
            RDMs[im] = np.einsum('ijk,ijk->ij', Ds[im],
                                 np.mean(Ds[idx], axis=0)) / P
        RDMs = np.mean(RDMs,axis=0)
        covRDMs[iRep] = np.cov(RDMs.transpose())
    
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    for iRep in range(Nrep):
        plt.plot(covRDMs[iRep][:],cov_pred[iRep][:], 'k.')
    plt.plot([np.min(covRDMs),np.max(covRDMs)],
             [np.min(covRDMs),np.max(covRDMs)],
             'k--')
    plt.title('ours', fontsize=24)
    plt.xlabel('sampled/true covariance', fontsize=15)
    plt.ylabel('formula covariance', fontsize=15)
    plt.gca().set_aspect('equal', adjustable='box')
    
    plt.subplot(1,2,2)
    for iRep in range(Nrep):
        plt.plot(covRDMs[iRep][:],covMatrix[iRep][:],'k.')
    plt.plot([np.min(covRDMs), np.max(covRDMs)],
             [np.min(covRDMs), np.max(covRDMs)],
             'k--')
    plt.title('Diedrichsen et. al.',fontsize=24)
    plt.xlabel('sampled/true covariance',fontsize = 15)
    plt.ylabel('formula covariance',fontsize = 15)
    plt.gca().set_aspect('equal', adjustable='box')
    
    plt.figure(figsize=(10,10))
    for iRep in range(Nrep):
        plt.plot(cov_pred[iRep][:],covMatrix[iRep][:],'k.')
    plt.plot([np.min(covRDMs), np.max(covRDMs)],
             [np.min(covRDMs), np.max(covRDMs)],
             'k--')
    plt.xlabel('ours',fontsize = 15)
    plt.ylabel('Diedrichsen et. al.',fontsize = 15)
    plt.gca().set_aspect('equal', adjustable='box')
    return covMatrix,cov_pred,covRDMs
    
    
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
    
   
def sampling_pool_RDM(U0,Nsubj,sigmaP,N=1000):
    return None 
    
def pipeline_pooling_dnn(model=None, Nvox=10, fname='alexnet', Nsubj=10,
                         Nrepeat=5, cross_residuals=True, shrinkage=0.4,
                         layer=3, Nsamp=3, Nstimuli=92):
    # runs the simulations to calculate a specific result for the pooling analysis
    # this produces 7 pooled RDMs for every sample:
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
    stimuli = stimuli[:Nstimuli]
    path = 'simulations/pooling_dnn'
    fname = path+'/'+fname
    if Nstimuli == 92:
        fnameTrue = fname+'True%d_%04d_%02dS_%02dR' % (
            layer, Nvox, Nsubj, Nrepeat)
        fnameSubj = fname+'Subj%d_%04d_%02dS_%02dR' % (
            layer, Nvox, Nsubj, Nrepeat)
        fnameSamples = fname+'Samp%d_%04d_%02dS_%02dR' % (
            layer, Nvox, Nsubj, Nrepeat)
    else:
        fnameTrue = fname+'True%d_%04d_%02d_%02dS_%02dR' % (
            layer, Nvox, Nstimuli, Nsubj, Nrepeat)
        fnameSubj = fname+'Subj%d_%04d_%02d_%02dS_%02dR' % (
            layer, Nvox, Nstimuli, Nsubj, Nrepeat)
        fnameSamples = fname+'Samp%d_%04d_%02d_%02dS_%02dR' % (
            layer, Nvox, Nstimuli, Nsubj, Nrepeat)
    repeats = 1
    duration = 5
    pause = 1
    endzeros = 20
    sd = np.array([5, 5])
    use_cor_noise = True
    sigma_noise = None
    resolution = None
    ar_coeff = None
    RDMs_true_subj = []
    RDMs_samples = []
    RDMs_covs = []
    for iSamp in tqdm.trange(Nsamp, position=0):
        RDM_true = dnn.get_true_RDM(model,layer,stimuli)
        RDM_true_subj = np.zeros((Nsubj,)+RDM_true.shape)
        RDM_samples = []
        covs11 = []
        covs12 = []
        covs13 = []
        covs21 = []
        covs22 = []
        covs23 = []
        total_dur = len(stimuli)*repeats*(duration+pause)+endzeros
        #print('\n starting simulations')
        for iSubj in tqdm.trange(Nsubj,position=1):
            Usubj, sigmaP, indices_space, weights = \
                dnn.get_sampled_representations(model,layer,sd,stimuli,Nvox)
            RDM_true_subj[iSubj] = pyrsa.calc_RDM_mahalanobis(Usubj,sigmaP)
            # replaced with proper sampling of a timecourse
            #Usamp = dnn.get_random_sample(Usubj,sigmaP,sigmaNoise,N) 
            rdm_samples_subj = np.zeros((len(stimuli),len(stimuli)))
            betas = np.zeros((Nrepeat,len(stimuli)+1,Nvox))
            timecourses = np.zeros((Nrepeat,total_dur,Nvox))
            designs = np.zeros((Nrepeat,total_dur,len(stimuli)+1))
            for iSamp in range(Nrepeat):
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
                betas[iSamp] = estimate_betas(design,timecourse)
                timecourses[iSamp] = timecourse
                designs[iSamp] = design
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
                sigma_p_est = np.eye(Nvox)
            rdm_samples_subj = pyrsa.calc_RDM_crossnobis(
                betas[:, :len(stimuli)], sigma_p_est)
            RDM_samples.append(rdm_samples_subj) 
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
        RDM_samples = np.array(RDM_samples)
        RDMs_true_subj.append(RDM_true_subj)
        RDM_pool = []
        cov_pool = []
        pool = pyrsa.pool_rdms(RDM_samples)
        RDM_pool.append(pyrsa.get_rdm_vector(pool))
        pool = pyrsa.pool_rdms(RDM_samples, covs11)
        RDM_pool.append(pool[0])
        cov_pool.append(pool[1])
        pool = pyrsa.pool_rdms(RDM_samples, covs12)
        RDM_pool.append(pool[0])
        cov_pool.append(pool[1])
        pool = pyrsa.pool_rdms(RDM_samples, covs13)
        RDM_pool.append(pool[0])
        cov_pool.append(pool[1])
        pool = pyrsa.pool_rdms(RDM_samples, covs21)
        RDM_pool.append(pool[0])
        cov_pool.append(pool[1])
        pool = pyrsa.pool_rdms(RDM_samples, covs22)
        RDM_pool.append(pool[0])
        cov_pool.append(pool[1])
        pool = pyrsa.pool_rdms(RDM_samples, covs23)
        RDM_pool.append(pool[0])
        cov_pool.append(pool[1])
        RDMs_samples.append(np.array(RDM_pool))
        RDMs_covs.append(cov_pool)
    RDMs_samples = np.array(RDMs_samples)
    RDMs_true_subj = np.array(RDMs_true_subj)
    np.save(fnameTrue, RDM_true)
    np.save(fnameSubj, RDMs_true_subj)
    np.save(fnameSamples, RDMs_samples)


def plot_dnn_pooling(fname='alexnet', Nstimuli=40, layer=3, Nsubj=10,
                     Nrepeat=5):
    path = 'simulations/pooling_dnn'
    fname = path+'/'+fname
    if Nstimuli == 92:
        fnameTrue = fname + 'True%d_%04d_%02dS_%02dR.npy' % (
            layer, 10, Nsubj, Nrepeat)
    else:
        fnameTrue = fname + 'True%d_%04d_%02d_%02dS_%02dR.npy' % (
            layer, 10, Nstimuli, Nsubj, Nrepeat)
    trueRDM = np.load(fnameTrue)
    error = np.zeros((5, 7))
    corr = np.zeros((5, 7))
    errorSubj = np.zeros((5))
    corrSubj = np.zeros((5))
    k = 0
    for Nvox in (10, 25, 75, 200, 500):
        if Nstimuli == 92:
            fnameSubj = fname + 'Subj%d_%04d_%02dS_%02dR.npy' % (
                layer, Nvox, Nsubj, Nrepeat)
            fnameSamples = fname + 'Samp%d_%04d_%02dS_%02dR.npy' % (
                layer, Nvox, Nsubj, Nrepeat)
        else:
            fnameSubj = fname + 'Subj%d_%04d_%02d_%02dS_%02dR.npy' % (
                layer, Nvox, Nstimuli, Nsubj, Nrepeat)
            fnameSamples = fname+'Samp%d_%04d_%02d_%02dS_%02dR.npy' % (
                layer, Nvox, Nstimuli, Nsubj, Nrepeat)
        trueSubj = np.load(fnameSubj)
        Nsamp = trueSubj.shape[0]
        trueSubj = trueSubj.reshape(Nsubj * Nsamp, Nstimuli, Nstimuli)
        samples = np.load(fnameSamples)
        errorSubj[k] = np.mean(pyrsa.error_rdm(trueSubj, trueRDM))
        corrSubj[k] = np.mean(pyrsa.corr_rdm(trueSubj, trueRDM))
        for iPool in range(7):
            error[k, iPool] = np.mean(pyrsa.error_rdm(samples[:, iPool],
                                                      trueRDM))
            corr[k, iPool] = np.mean(pyrsa.corr_rdm(samples[:, iPool],
                                                    trueRDM))
        k = k+1
    plt.figure()
    plt.plot(corr)
    plt.xticks(ticks=[0, 1, 2, 3, 4], labels=['10', '25', '75', '200', '500'])
    plt.xlabel('Number of Voxels')
    plt.ylabel('RDM correlation')
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
