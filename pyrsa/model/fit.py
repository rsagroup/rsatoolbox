#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fitter functions
@author: heiko
"""

import numpy as np
from pyrsa.model import ModelSelect
from pyrsa.rdm import compare


def fit_mock(model, data, method='cosine', pattern_sample=None,
             pattern_select=None, pattern_descriptor=None):
    """ formally acceptable fitting method which always returns a vector of
    zeros
    
    Args:
        model(pyrsa.model.Model): model to be fit
        data(pyrsa.rdm.RDMs): Data to fit to
        method(String): Evaluation method
        pattern_sample(numpy.ndarray): Which patterns are sampled
        pattern_select(list of String): pattern keys
        pattern_descriptor(String): Which descriptor is used
        
    Returns:
        theta(numpy.ndarray): parameter vector

    """
    return np.zeros(model.n_param)


def fit_select(model, data, method='cosine', pattern_sample=None,
               pattern_select=None, pattern_descriptor=None):
    """ fits selection models by evaluating each rdm and selcting the one
    with best performance. Works only for ModelSelect
    
    Args:
        model(pyrsa.model.Model): model to be fit
        data(pyrsa.rdm.RDMs): Data to fit to
        method(String): Evaluation method
        pattern_sample(numpy.ndarray): Which patterns are sampled
        pattern_select(list of String): pattern keys
        pattern_descriptor(String): Which descriptor is used
        
    Returns:
        theta(int): parameter vector

    """
    assert isinstance(model, ModelSelect)
    evaluations = np.zeros(model.n_rdm)
    for i_rdm in range(model.n_rdm):
        pred = model.predict_rdm(i_rdm)
        evaluations[i_rdm] = np.mean(compare(pred, data))
    theta = np.argmax(evaluations)
    return theta
