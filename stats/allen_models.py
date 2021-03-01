#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created 

@author: baihan
"""

import numpy as np
import tqdm
import pyrsa
from scipy.ndimage import gaussian_filter as gaussian_filter

def get_allen_RDM(model, layer, stimuli, method='euclidean', smoothing=None,
                 average=False):
    U = list([])
    for istimulus in stimuli:
        Ustim = get_complete_representation(model=model, layer=layer,
                                            stimulus=istimulus)
        if average:
            Ustim = np.sum(Ustim, axis=1, keepdims=True)
        if smoothing:
            if np.isfinite(smoothing):
                sd = np.array(Ustim.shape[2:4]) * smoothing
                Ustim = gaussian_filter(Ustim, [0, 0, sd[0], sd[1]])
            elif smoothing == np.inf:
                Ustim = np.average(np.average(Ustim, axis=3), axis=2)
        U.append(Ustim.flatten())
    U = np.array(U)
    data = pyrsa.data.Dataset(U)
    if not method == 'euclidean':
        return pyrsa.rdm.calc_rdm(data, method=method)
    else:
        return pyrsa.rdm.calc_rdm_euclid_save_memory(data)
    
def get_allen_models(model_type, stimuli,
               n_layer=12, smoothing=None):
    """
    creates the models for the Allen's different ROIs
    most commonly used model is fixed_average, which produces the correct
    weighting between full and feature averaged RDM at a given smoothing
    other model types described below.
    All models adapt to the smoothing of the voxels

    Parameters
    ----------
    model_type : String
        which type of model to create
        'fixed_full' : fixed model of full feature space RDM
        'fixed_average' : correct weighting of fixed_full and fixed_mean
        'fixed_mean' : fixed model of feature averaged maps
        'select_full' : selection model of full feature space
        'select_mean' : selection model of feature averaged space
        'select_average' : selection model for correct weighting
        'select_both' : selection among both _full and _mean
        'interpolate_full' : interpolation model for full feature space
        'interpolate_mean' : interpolation model for feature averaged space
        'interpolate_average' : interpolation model both _full and _mean
        'interpolate_both' : interpolation model both _full and _mean
        'weighted_avgfull' : weighted model of 4 rdms full and mean for
            zero and infinite smoothing
        
    stimuli : images
        model inputs for conditions
    n_layer : int, optional
        number of layers -> how many models? The default is 12.
    smoothing : float, optional
        how much smoothing to apply for models. The default is no smoothing.

    Returns
    -------
    models : list of pyrsa.model.Model
        Models corresponding to the model possibilities 

    """
    n_stimuli = len(stimuli)
    pat_desc = {'stim': np.arange(n_stimuli)}
    models = []
    smoothings = np.array([0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, np.inf])
    for i_layer in tqdm.trange(n_layer):
        if model_type == 'fixed_full':
            rdm = get_allen_RDM(
                model='allen_roi',
                layer=i_layer,
                stimuli=stimuli,
                smoothing=smoothing)
            rdm.pattern_descriptors = pat_desc
            model = pyrsa.model.ModelFixed('Layer%02d' % i_layer, rdm)
        elif model_type == 'fixed_average':
            rdm1 = get_allen_RDM(
                model='allen_roi',
                layer=i_layer,
                stimuli=stimuli,
                smoothing=smoothing)
            rdm2 = get_allen_RDM(
                model='allen_roi',
                layer=i_layer,
                stimuli=stimuli,
                smoothing=smoothing,
                average=True)
            # this weighting comes from E(U[0,1]) **2 = 0.25
            # and Var(U[0,1]) = 1/12
            # Thus 1 : 3 should be the right weighting between the two
            # euclidean distances
            rdm = pyrsa.rdm.RDMs(3 * rdm1.get_vectors() + rdm2.get_vectors(),
                                 pattern_descriptors=pat_desc)
            model = pyrsa.model.ModelFixed('Layer%02d' % i_layer, rdm)
        elif model_type == 'fixed_mean':
            rdm2 = get_allen_RDM(
                model='allen_roi',
                layer=i_layer,
                stimuli=stimuli,
                smoothing=smoothing,
                average=True)
            rdm = pyrsa.rdm.RDMs(rdm2.get_vectors(),
                                 pattern_descriptors=pat_desc)
            model = pyrsa.model.ModelFixed('Layer%02d' % i_layer, rdm)
        elif model_type == 'select_full':
            rdms = []
            for i_smooth, smooth in enumerate(smoothings):
                rdm = get_allen_RDM(
                    model='allen_roi',
                    layer=i_layer,
                    stimuli=stimuli,
                    smoothing=smooth,
                    average=False)
                rdm.pattern_descriptors = pat_desc
                rdms.append(rdm)
            rdms = pyrsa.rdm.concat(rdms)
            model = pyrsa.model.ModelSelect('Layer%02d' % i_layer, rdms)
        elif model_type == 'select_mean':
            rdms = []
            for i_smooth, smooth in enumerate(smoothings):
                rdm = get_allen_RDM(
                    model='allen_roi',
                    layer=i_layer,
                    stimuli=stimuli,
                    smoothing=smooth,
                    average=True)
                rdm.pattern_descriptors = pat_desc
                rdms.append(rdm)
            rdms = pyrsa.rdm.concat(rdms)
            model = pyrsa.model.ModelSelect('Layer%02d' % i_layer, rdms)
        elif model_type == 'select_both':
            rdms = []
            for i_smooth, smooth in enumerate(smoothings):
                rdm = get_allen_RDM(
                    model='allen_roi',
                    layer=i_layer,
                    stimuli=stimuli,
                    smoothing=smooth,
                    average=False)
                rdm.pattern_descriptors = pat_desc
                rdms.append(rdm)
                rdm = get_allen_RDM(
                    model='allen_roi',
                    layer=i_layer,
                    stimuli=stimuli,
                    smoothing=smooth,
                    average=True)
                rdm.pattern_descriptors = pat_desc
                rdms.append(rdm)
            rdms = pyrsa.rdm.concat(rdms)
            model = pyrsa.model.ModelSelect('Layer%02d' % i_layer, rdms)
        elif model_type == 'select_average':
            rdms = []
            for i_smooth, smooth in enumerate(smoothings):
                rdm1 = get_allen_RDM(
                    model='allen_roi',
                    layer=i_layer,
                    stimuli=stimuli,
                    smoothing=smooth,
                    average=False)
                rdm2 = get_allen_RDM(
                    model='allen_roi',
                    layer=i_layer,
                    stimuli=stimuli,
                    smoothing=smooth,
                    average=True)
                rdm = pyrsa.rdm.RDMs(3 * rdm1.get_vectors() 
                                     + rdm2.get_vectors(),
                                     pattern_descriptors=pat_desc)
                rdms.append(rdm)
            rdms = pyrsa.rdm.concat(rdms)
            model = pyrsa.model.ModelSelect('Layer%02d' % i_layer, rdms)
        elif model_type == 'interpolate_full':
            rdms = []
            for i_smooth, smooth in enumerate(smoothings):
                rdm = get_allen_RDM(
                    model='allen_roi',
                    layer=i_layer,
                    stimuli=stimuli,
                    smoothing=smooth,
                    average=False)
                rdm.pattern_descriptors = pat_desc
                rdms.append(rdm)
            rdms = pyrsa.rdm.concat(rdms)
            model = pyrsa.model.ModelInterpolate('Layer%02d' % i_layer, rdms)
        elif model_type == 'interpolate_mean':
            rdms = []
            for i_smooth, smooth in enumerate(smoothings):
                rdm = get_allen_RDM(
                    model='allen_roi',
                    layer=i_layer,
                    stimuli=stimuli,
                    smoothing=smooth,
                    average=True)
                rdm.pattern_descriptors = pat_desc
                rdms.append(rdm)
            rdms = pyrsa.rdm.concat(rdms)
            model = pyrsa.model.ModelInterpolate('Layer%02d' % i_layer, rdms)
        elif model_type == 'interpolate_both':
            rdms = []
            for i_smooth, smooth in enumerate(smoothings):
                rdm = get_allen_RDM(
                    model='allen_roi',
                    layer=i_layer,
                    stimuli=stimuli,
                    smoothing=smooth,
                    average=True)
                rdm.pattern_descriptors = pat_desc
                rdms.append(rdm)
            for i_smooth in range(len(smoothings) - 1, -1, -1):
                rdm = get_allen_RDM(
                    model='allen_roi',
                    layer=i_layer,
                    stimuli=stimuli,
                    smoothing=smoothings[i_smooth],
                    average=False)
                rdm.pattern_descriptors = pat_desc
                rdms.append(rdm)
            rdms = pyrsa.rdm.concat(rdms)
            model = pyrsa.model.ModelInterpolate('Layer%02d' % i_layer, rdms)
        elif model_type == 'interpolate_average':
            rdms = []
            for i_smooth, smooth in enumerate(smoothings):
                rdm1 = get_allen_RDM(
                    model='allen_roi',
                    layer=i_layer,
                    stimuli=stimuli,
                    smoothing=smooth,
                    average=False)
                rdm2 = get_allen_RDM(
                    model='allen_roi',
                    layer=i_layer,
                    stimuli=stimuli,
                    smoothing=smooth,
                    average=True)
                rdm = pyrsa.rdm.RDMs(3 * rdm1.get_vectors() 
                                     + rdm2.get_vectors(),
                                     pattern_descriptors=pat_desc)
                rdms.append(rdm)
            rdms = pyrsa.rdm.concat(rdms)
            model = pyrsa.model.ModelSelect('Layer%02d' % i_layer, rdms)
        elif model_type == 'weighted_avgfull':
            rdms = []
            rdm = get_allen_RDM(
                model='allen_roi',
                layer=i_layer,
                stimuli=stimuli,
                smoothing=None,
                average=True)
            rdms.append(rdm)
            rdm = get_allen_RDM(
                model='allen_roi',
                layer=i_layer,
                stimuli=stimuli,
                smoothing=None,
                average=False)
            rdms.append(rdm)
            rdm = get_allen_RDM(
                model='allen_roi',
                layer=i_layer,
                stimuli=stimuli,
                smoothing=np.inf,
                average=True)
            rdms.append(rdm)
            rdm = get_allen_RDM(
                model='allen_roi',
                layer=i_layer,
                stimuli=stimuli,
                smoothing=np.inf,
                average=False)
            rdms.append(rdm)
            rdms = pyrsa.rdm.concat(rdms)
            model = pyrsa.model.ModelWeighted('Layer%02d' % i_layer, rdms)
            model.default_fitter = pyrsa.model.fitter.fit_regress
        models.append(model)
    return models