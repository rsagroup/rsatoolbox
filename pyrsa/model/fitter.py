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
    print(evaluations)
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


def loss_weighted_derivative(w,candidateRDMs, trueRDM,method='euclid',cov=None):
    w = w.reshape((candidateRDMs.shape[0],1))
    RDM = np.sum(w**2*candidateRDMs,axis=0)
    if method == 'euclid':
        if len(trueRDM.shape)==1:
            jac = np.sum(2*(RDM-trueRDM)*2*w*candidateRDMs,axis=-1)
        elif len(trueRDM.shape)==2:
            jac = np.einsum('ij,kj->k',2*(RDM-trueRDM)*2,w*candidateRDMs)
    elif method == 'cosine':
        if len(trueRDM.shape)==1:
            dnorm = np.sum(RDM*candidateRDMs,axis=1)
            b = trueRDM/np.linalg.norm(trueRDM)
            a = RDM/np.linalg.norm(RDM)
            jac1 = np.sum(b*candidateRDMs,axis=1)/np.linalg.norm(RDM)
            jacnorm = np.sum(b*a)*dnorm/(np.linalg.norm(RDM))/(np.linalg.norm(RDM))
            dnorm2 = 0.02*((np.linalg.norm(RDM))-1)*dnorm/(np.linalg.norm(RDM))
            jac = 2*w.flatten()*(jac1-jacnorm+dnorm2)
        else:
            dnorm = np.sum(RDM*candidateRDMs,axis=1)
            b = trueRDM/np.linalg.norm(trueRDM,axis=1,keepdims=True)
            a = RDM/np.linalg.norm(RDM)
            jac1 = np.mean(np.einsum('ij,kj->ik',b,candidateRDMs)/np.linalg.norm(RDM),axis=0)
            jacnorm = np.sum(b*a)*dnorm/(np.linalg.norm(RDM))/(np.linalg.norm(RDM))/b.shape[0]
            dnorm2 = 0.02*((np.linalg.norm(RDM))-1)*dnorm/(np.linalg.norm(RDM))
            jac = 2*w.flatten()*(jac1-jacnorm+dnorm2)
    elif method == 'likelihood':
        assert not cov is None, 'covariance has to be given for likelihood'
        for i in range(cov.shape[0]):
            cov[i] = np.linalg.inv(cov[i])
        if len(trueRDM.shape)==1:
            jac = 2*w.flatten()*np.matmul(candidateRDMs,np.matmul(RDM-trueRDM,(cov+cov.T)))
        elif len(trueRDM.shape)==2:
            jac = np.einsum('kj,ij->k',candidateRDMs,np.einsum('ij,ijl->il',RDM-trueRDM,2*cov))
    return jac