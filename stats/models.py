#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 15 07:57:55 2020

@author: heiko
"""

import numpy as np
import tqdm
import nn_simulations as dnn
import pyrsa


def get_models(model_type, stimuli,
               n_layer=12, n_sim=1000, smoothing=None):
    n_stimuli = len(stimuli)
    pat_desc = {'stim': np.arange(n_stimuli)}
    models = []
    smoothings = np.array([0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, np.inf])
    dnn_model = dnn.get_default_model()
    for i_layer in tqdm.trange(n_layer):
        if model_type == 'fixed_full':
            rdm = dnn.get_true_RDM(
                model=dnn_model,
                layer=i_layer,
                stimuli=stimuli,
                smoothing=smoothing)
            rdm.pattern_descriptors = pat_desc
            model = pyrsa.model.ModelFixed('Layer%02d' % i_layer, rdm)
        elif model_type == 'fixed_average':
            rdm1 = dnn.get_true_RDM(
                model=dnn_model,
                layer=i_layer,
                stimuli=stimuli,
                smoothing=smoothing)
            rdm2 = dnn.get_true_RDM(
                model=dnn_model,
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
        elif model_type == 'select_full':
            rdms = []
            for i_smooth, smooth in enumerate(smoothings):
                rdm = dnn.get_true_RDM(
                    model=dnn_model,
                    layer=i_layer,
                    stimuli=stimuli,
                    smoothing=smooth,
                    average=False)
                rdm.pattern_descriptors = pat_desc
                rdms.append(rdm)
            rdms = pyrsa.rdm.concat(rdms)
            model = pyrsa.model.ModelSelect('Layer%02d' % i_layer, rdms)
        elif model_type == 'select_avg':
            rdms = []
            for i_smooth, smooth in enumerate(smoothings):
                rdm = dnn.get_true_RDM(
                    model=dnn_model,
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
                rdm = dnn.get_true_RDM(
                    model=dnn_model,
                    layer=i_layer,
                    stimuli=stimuli,
                    smoothing=smooth,
                    average=False)
                rdm.pattern_descriptors = pat_desc
                rdms.append(rdm)
                rdm = dnn.get_true_RDM(
                    model=dnn_model,
                    layer=i_layer,
                    stimuli=stimuli,
                    smoothing=smooth,
                    average=True)
                rdm.pattern_descriptors = pat_desc
                rdms.append(rdm)
            rdms = pyrsa.rdm.concat(rdms)
            model = pyrsa.model.ModelSelect('Layer%02d' % i_layer, rdms)
        elif model_type == 'interpolate_full':
            rdms = []
            for i_smooth, smooth in enumerate(smoothings):
                rdm = dnn.get_true_RDM(
                    model=dnn_model,
                    layer=i_layer,
                    stimuli=stimuli,
                    smoothing=smooth,
                    average=False)
                rdm.pattern_descriptors = pat_desc
                rdms.append(rdm)
            rdms = pyrsa.rdm.concat(rdms)
            model = pyrsa.model.ModelInterpolate('Layer%02d' % i_layer, rdms)
        elif model_type == 'interpolate_avg':
            rdms = []
            for i_smooth, smooth in enumerate(smoothings):
                rdm = dnn.get_true_RDM(
                    model=dnn_model,
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
                rdm = dnn.get_true_RDM(
                    model=dnn_model,
                    layer=i_layer,
                    stimuli=stimuli,
                    smoothing=smooth,
                    average=True)
                rdm.pattern_descriptors = pat_desc
                rdms.append(rdm)
            for i_smooth in range(len(smoothings) - 1, -1, -1):
                rdm = dnn.get_true_RDM(
                    model=dnn_model,
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
                model=dnn_model,
                layer=i_layer,
                stimuli=stimuli,
                smoothing=None,
                average=True)
            rdms.append(rdm)
            rdm = dnn.get_true_RDM(
                model=dnn_model,
                layer=i_layer,
                stimuli=stimuli,
                smoothing=None,
                average=False)
            rdms.append(rdm)
            rdm = dnn.get_true_RDM(
                model=dnn_model,
                layer=i_layer,
                stimuli=stimuli,
                smoothing=np.inf,
                average=True)
            rdms.append(rdm)
            rdm = dnn.get_true_RDM(
                model=dnn_model,
                layer=i_layer,
                stimuli=stimuli,
                smoothing=np.inf,
                average=False)
            rdms.append(rdm)
            model = pyrsa.model.ModelWeighted('Layer%02d' % i_layer, rdms)
        models.append(model)
    return models
