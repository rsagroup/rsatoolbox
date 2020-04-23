#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 15:54:31 2020

@author: heiko
"""

import numpy as np
import scipy.optimize as opt
from pyrsa.rdm import compare


def fit_mock(model, data, method='cosine', pattern_sample=None,
             pattern_descriptor=None):
    """ formally acceptable fitting method which always returns a vector of
    zeros

    Args:
        model(pyrsa.model.Model): model to be fit
        data(pyrsa.rdm.RDMs): Data to fit to
        method(String): Evaluation method
        pattern_sample(numpy.ndarray): Which patterns are sampled
        pattern_descriptor(String): Which descriptor is used

    Returns:
        theta(numpy.ndarray): parameter vector

    """
    return np.zeros(model.n_param)


def fit_select(model, data, method='cosine', pattern_sample=None,
               pattern_descriptor=None):
    """ fits selection models by evaluating each rdm and selcting the one
    with best performance. Works only for ModelSelect

    Args:
        model(pyrsa.model.Model): model to be fit
        data(pyrsa.rdm.RDMs): Data to fit to
        method(String): Evaluation method
        pattern_sample(numpy.ndarray): Which patterns are sampled
        pattern_descriptor(String): Which descriptor is used

    Returns:
        theta(int): parameter vector

    """
    evaluations = np.zeros(model.n_rdm)
    for i_rdm in range(model.n_rdm):
        pred = model.predict_rdm(i_rdm)
        if not (pattern_sample is None or pattern_descriptor is None):
            pred = pred.subsample_pattern(pattern_descriptor, pattern_sample)
        evaluations[i_rdm] = np.mean(compare(pred, data, method=method))
    theta = np.argmin(evaluations)
    return theta


def fit_optimize(model, data, method='cosine', pattern_sample=None,
                 pattern_descriptor=None):
    """
    fitting theta using optimization
    currently allowed for ModelWeighted only

    Args:
        model(Model): the model to be fit
        data(pyrsa.rdm.RDMs): data to be fit
        method(String, optional): evaluation metric The default is 'cosine'.
        pattern_sample(numpy.ndarray, optional)
            sampled patterns The default is None.
        pattern_descriptor (String, optional)
            descriptor used for fitting. The default is None.

    Returns:
        numpy.ndarray: theta, parameter vector for the model

    """
    def _loss_opt(theta):
        return _loss(theta, model, data, method=method,
                     pattern_sample=pattern_sample,
                     pattern_descriptor=pattern_descriptor)
    theta0 = np.random.rand(model.n_param)
    theta = opt.minimize(_loss_opt, theta0)
    return theta.x


def fit_interpolate(model, data, method='cosine', pattern_sample=None,
                 pattern_descriptor=None):
    """
    fitting theta using bisection optimization
    allowed for ModelInterpolate only

    Args:
        model(Model): the model to be fit
        data(pyrsa.rdm.RDMs): data to be fit
        method(String, optional): evaluation metric The default is 'cosine'.
        pattern_sample(numpy.ndarray, optional)
            sampled patterns The default is None.
        pattern_descriptor (String, optional)
            descriptor used for fitting. The default is None.

    Returns:
        numpy.ndarray: theta, parameter vector for the model

    """
    results = []
    for i_pair in range(model.n_rdm-1):
        def loss_opt(w):
            theta = np.zeros(model.n_param)
            theta[i_pair] = w
            theta[i_pair+1] = 1-w
            return _loss(theta, model, data, method=method,
                     pattern_sample=pattern_sample,
                     pattern_descriptor=pattern_descriptor)
        results.append(
            opt.minimize_scalar(loss_opt, np.array([.5]),
                                method='bounded', bounds = (0,1)))
    losses = [r.fun for r in results]
    i_pair = np.argmin(losses)
    result = results[i_pair]
    theta = np.zeros(model.n_rdm)
    theta[i_pair] = result.x
    theta[i_pair+1] = 1-result.x
    return theta

def _loss(theta, model, data, method='cosine', cov=None,
          pattern_descriptor=None, pattern_sample=None):
    """Method for calculating a loss for a model and parameter combination

    Args:
        theta(numpy.ndarray): evaluated parameter value
        model(Model): the model to be fit
        data(pyrsa.rdm.RDMs): data to be fit
        method(String, optional): evaluation metric The default is 'cosine'.
        pattern_sample(numpy.ndarray, optional)
            sampled patterns The default is None.
        pattern_descriptor (String, optional)
            descriptor used for fitting. The default is None.
        cov(numpy.ndarray, optional):
            Covariance matrix for likelihood based evaluation. 
            It is ignored otherwise. The default is None.

    Returns:
        
        numpy.ndarray: loss

    """
    pred = model.predict_rdm(theta)
    if not (pattern_sample is None or pattern_descriptor is None):
        pred = pred.subsample_pattern(pattern_descriptor, pattern_sample)
    return np.mean(compare(pred, data, method=method))
