#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 11:09:09 2019

@author: heiko

This file contains the methods to simulate responses from neural networks. 
This requires pytorch and torchvision

If you run functions in here without further input they will be run on alexnet,
whose weights will be downloaded if you don't have them already. 
"""

import torch
import torchvision
import PIL 
import numpy as np
from scipy.ndimage import gaussian_filter as gaussian_filter
import scipy.signal as signal
import tqdm
import os
from hrf import spm_hrf
import pyrsa

# initial transormation expected by all torchvision models
normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

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


def generate_design_random(nStimuli, repeats=3, duration=5, pause=0, endzeros=20):
    # generates a design matrix for a given number of stimuli presented in random order 
    # each stimulus is shown before any repetitions happen
    # duration and pause are measured in measurement cycles
    design = np.zeros((nStimuli*repeats*(duration+pause)+endzeros,nStimuli+1))
    design[:,-1] = 1
    shape = np.concatenate((np.ones(duration),np.zeros(pause)))
    length = len(shape)
    lengthAll = nStimuli*(duration+pause)
    for iRep in range(repeats):
        stimOrder = np.arange(nStimuli)
        np.random.shuffle(stimOrder)
        for iStim in range(nStimuli):
            design[(length*iStim+lengthAll*iRep):(length*(iStim+1)+lengthAll*iRep),stimOrder[iStim]]=1
    return design


def generate_timecourse(design, U0, sigma_noise=1, hrf=None,
                        resolution=None, ar_coeff=None, sigmaP = None):
    # generates a timecourse with random normal noise added after the convolution with the hrf
    if ar_coeff is None:
        ar_coeff = .5
    if resolution is None:
        resolution = 2
    if hrf is None:
        t = np.arange(0,40,resolution)
        hrf = resolution*spm_hrf(t)
    if sigmaP is None:
        sigmaP = np.eye(U0.shape[1])
    hrf = np.array([hrf]).transpose()
    data = design  
    data = signal.convolve(design,hrf,mode = 'full')[:design.shape[0]]
    Ubias = np.concatenate((U0,np.zeros((1,U0.shape[1]))),axis = 0)
    data = np.einsum('ki,ij->kj',data,Ubias)
    noise = np.random.randn(data.shape[0],data.shape[1])
    for iNoise in range(1,noise.shape[0]):
        noise[iNoise] = ar_coeff*noise[iNoise-1] + np.sqrt((1-ar_coeff**2))*noise[iNoise] 
    noise = np.matmul(np.linalg.cholesky(sigmaP),noise.T).T
    data = data + sigma_noise * noise
    return data


def get_random_sample(U, sigmaP, sigmaNoise=None, N=10):
    if sigmaNoise == 0 or sigmaNoise is None:
        if len(U.shape)==1:
            rand1 = np.random.randn(N,U.shape[0])
        elif len(U.shape) == 2:
            rand1 = np.random.randn(N,U.shape[0],U.shape[1])
        elif len(U.shape) == 3:
            rand1 = np.random.randn(N,U.shape[0],U.shape[1],U.shape[2])
        Usamp = U + np.matmul(rand1,np.linalg.cholesky(sigmaP).transpose())
    elif not isinstance(sigmaNoise,np.ndarray):
        if len(U.shape)==1:
            rand1 = np.random.randn(N,U.shape[0])
            rand2 = np.random.randn(N,U.shape[0])
        elif len(U.shape) == 2:
            rand1 = np.random.randn(N,U.shape[0],U.shape[1])
            rand2 = np.random.randn(N,U.shape[0],U.shape[1])
        elif len(U.shape) == 3:
            rand1 = np.random.randn(N,U.shape[0],U.shape[1],U.shape[2])
            rand2 = np.random.randn(N,U.shape[0],U.shape[1],U.shape[2])
        Usamp = U + np.matmul(rand1,np.linalg.cholesky(sigmaP).transpose()) + sigmaNoise*rand2
    elif isinstance(sigmaNoise,np.ndarray):
        if len(U.shape)==1:
            rand1 = np.random.randn(N,U.shape[0])
            rand2 = np.random.randn(N,U.shape[0])
        elif len(U.shape) == 2:
            rand1 = np.random.randn(N,U.shape[0],U.shape[1])
            rand2 = np.random.randn(N,U.shape[0],U.shape[1])
        elif len(U.shape) == 3:
            rand1 = np.random.randn(N,U.shape[0],U.shape[1],U.shape[2])
            rand2 = np.random.randn(N,U.shape[0],U.shape[1],U.shape[2])
        Usamp = (U + np.matmul(rand1,np.linalg.cholesky(sigmaP).transpose()) + 
                     np.matmul(rand2,np.linalg.cholesky(sigmaNoise).transpose()))
    return Usamp


def get_true_RDM(model, layer, stimuli, method='euclidean', smoothing=None,
                 average=False):
    U = list([])
    for istimulus in stimuli:
        Ustim = get_complete_representation(model=model, layer=layer,
                                            stimulus=istimulus)
        if average:
            Ustim = np.mean(Ustim, axis=1, keepdims=True)
        if smoothing:
            if np.isfinite(smoothing):
                sd  = np.array(Ustim.shape[2:4]) * smoothing
                Ustim = gaussian_filter(Ustim,[0, 0, sd[0], sd[1]])
            elif smoothing == np.inf:
                Ustim = np.average(np.average(Ustim, axis=3), axis=2)
        U.append(Ustim.flatten())
    U = np.array(U)
    data = pyrsa.data.Dataset(U)
    return pyrsa.rdm.calc_rdm(data, method=method)


def get_sampled_representations(model, layer, sd, stimList, N):
    U = get_complete_representation(model=model, layer=layer,
                                    stimulus=stimList[0])
    indices_space, weights = get_random_indices_conv(U.shape,N)
    sigmaP = get_sampled_sigmaP(U.shape,indices_space,weights,sd)
    U = [sample_representation(np.squeeze(U),indices_space,weights,sd)]
    for stimulus in stimList[1:]:
        Ustim = get_complete_representation(model=model, layer=layer,
                                            stimulus=stimulus)
        U.append(sample_representation(np.squeeze(Ustim), indices_space,
                                       weights, sd))
    return (np.array(U), sigmaP, indices_space, weights)


def get_sampled_representation_random(model, layer, sd, stimulus, N):
    U = get_complete_representation(model=model, layer=layer,
                                    stimulus=stimulus)
    indices_space, weights = get_random_indices_conv(U.shape,N)
    sigmaP = get_sampled_sigmaP(U.shape,indices_space, weights, sd)
    U = sample_representation(np.squeeze(U), indices_space, weights, sd)
    return (U, sigmaP, indices_space, weights)


def get_sampled_representation(indices_space, weights, sd,
                               model=None, layer=0, stimulus=None):
    U = get_complete_representation(model=model, layer=layer,
                                    stimulus=stimulus)
    U = sample_representation(U, indices_space=indices_space,
                              weights=weights, sd=sd)
    return U


def sample_representation(U, indices_space, weights, sd):
    sd = sd * U.shape[2]
    if len(U.shape)==3:
        U = gaussian_filter(U, [0, sd[0], sd[1]])
        U = U[:,indices_space[0], indices_space[1]]
        U = np.einsum('in,in->n', U, weights)
    elif len(U.shape)==4:
        U = gaussian_filter(U, [0, 0, sd[0], sd[1]])
        U = U[:, :, indices_space[0], indices_space[1]]
        U = np.einsum('kin,in->kn', U, weights)
    return U


def get_sampled_sigmaP(Ushape, indices_space, weights, sd):
    if len(Ushape)==4:
        Ushape = Ushape[1:]
    stack = np.zeros((indices_space.shape[1], Ushape[1], Ushape[2]))
    for iC in range(indices_space.shape[1]):
        stack[iC, indices_space[0,iC], indices_space[1,iC]]=1
    stack = gaussian_filter(stack,[0, sd[0], sd[1]])
    covGauss = np.einsum('nij,kij->nk',stack,stack)/stack.shape[1]/stack.shape[2]
    covWeights = np.einsum('in,ik->nk',weights,weights)/weights.shape[0]
    sigmaP = covGauss * covWeights
    sigmaP = sigmaP / np.max(sigmaP)
    return sigmaP


def get_complete_representation(model=None, layer=0, stimulus=None):
    if model is None:
        model = get_default_model()
    if stimulus is None:
        stimulus = torch.rand([1, 3, 224, 224])
    elif isinstance(stimulus, PIL.Image.Image):
        stimulus = stimulus.resize((224, 224))
        stimulus = np.array(stimulus).transpose(2,0,1)
        if stimulus.shape[0] == 4:
            stimulus = stimulus[:3]
        stimulus = torch.tensor(stimulus,dtype=torch.float)/255
        stimulus = normalize(stimulus)
        stimulus = torch.unsqueeze(stimulus,0)
    model = model.eval()
    U = model.features[:layer](stimulus)
    if layer > len(model.features):
        U = U.view(U.shape[0],-1)
        U = model.classifier[:(layer-len(model.features))](U)
    return U.detach().numpy()


def get_random_indices_conv(Ushape, N):
    # gets N randomly placed indices for a representation with shape U
    # for convolutional layers-> xpos, ypos & features weights
    # weights are uniform in [0,1]
    # position is uniform in space
    if len(Ushape) == 4:
        Ushape = Ushape[1:]
    weights = np.random.rand(Ushape[0], N)
    indx = np.random.randint(0, Ushape[1], N)
    indy = np.random.randint(0, Ushape[2], N)
    indices_space = np.array([indx,indy])
    return indices_space, weights


def get_default_model():
    return torchvision.models.alexnet(pretrained=True)




