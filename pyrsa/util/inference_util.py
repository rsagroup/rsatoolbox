#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 14:56:15 2020

@author: heiko
"""

import numpy as np
from scipy.stats import rankdata
from pyrsa.model import Model
from pyrsa.rdm import RDMs
from collections.abc import Iterable


def input_check_model(model, theta, fitter=None, N=1):
    if isinstance(model, Model):
        evaluations = np.zeros(N)
    elif isinstance(model, Iterable):
        if N > 1:
            evaluations = np.zeros((N,len(model)))
        else:
            evaluations = np.zeros(len(model))
        if not theta is None:
            assert isinstance(theta, Iterable), 'If a list of models is' \
                + ' passed theta must be a list of parameters'
            assert len(model) == len(theta), 'there should equally many' \
                + ' models as parameters'
        else:
            theta = [None] * len(model)
        if fitter is None:
            fitter = [None] * len(model)
        else:
            assert len(fitter) == len(model), 'if fitters are passed ' \
                + 'there should be as many as models'
        for k in range(len(model)):
            if fitter[k] is None:
                fitter[k] = model[k].default_fitter
    else:
        raise ValueError('model should be a pyrsa.model.Model or a list of'
                         + ' such objects')
    return evaluations, theta, fitter


def pool_rdm(rdms, method='cosine'):
    """pools multiple RDMs into the one with maximal performance under a given
    evaluation metric
    rdm_descriptors of the generated rdms are empty

    Args:
        rdms (pyrsa.rdm.RDMs):
            RDMs to be pooled
        method : String, optional
            Which comparison method to optimize for. The default is 'cosine'.

    Returns:
        pyrsa.rdm.RDMs: the pooled RDM, i.e. a RDM with maximal performance
            under the chosen method

    """
    rdm_vec = rdms.get_vectors()
    if method == 'euclid':
        rdm_vec = np.mean(rdm_vec, axis=0, keepdims=True)
    elif method == 'cosine':
        rdm_vec = rdm_vec/np.mean(rdm_vec, axis=1, keepdims=True)
        rdm_vec = np.mean(rdm_vec, axis=0, keepdims=True)
    elif method == 'corr':
        rdm_vec = rdm_vec - np.mean(rdm_vec, axis=1, keepdims=True)
        rdm_vec = rdm_vec / np.std(rdm_vec, axis=1, keepdims=True)
        rdm_vec = np.mean(rdm_vec, axis=0, keepdims=True)
        rdm_vec = rdm_vec - np.min(rdm_vec)
    elif method == 'spearman':
        rdm_vec = np.array([rankdata(v) for v in rdm_vec])
        rdm_vec = np.mean(rdm_vec, axis=0, keepdims=True)
    elif method == 'kendall':
        raise NotImplementedError('pooling for ranks not yet implemented!')
    else:
        raise ValueError('Unknown RDM comparison method requested!')
    return RDMs(rdm_vec,
                dissimilarity_measure=rdms.dissimilarity_measure,
                descriptors=rdms.descriptors,
                rdm_descriptors=None,
                pattern_descriptors=rdms.pattern_descriptors)
