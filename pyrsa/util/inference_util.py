#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inference module utilities
"""

import numpy as np
from pyrsa.model import Model
from collections.abc import Iterable


def input_check_model(model, theta=None, fitter=None, N=1):
    """ Checks whether model related inputs to evaluations are valid and
    generates an evaluation-matrix of fitting size.

    Args:
        model : [list of] pyrsa.rdm.RDMs
            the models to be evaluated
        theta : numpy.ndarray or list , optional
            Parameter(s) for the model(s). The default is None.
        fitter : [list of] function, optional
            fitting function to overwrite the model default.
            The default is None, i.e. keep default
        N : int, optional
            number of samples/rows in evaluations matrix. The default is 1.

    Returns:
        evaluations : numpy.ndarray
            empty evaluations-matrix
        theta : list
            the processed and checked model parameters
        fitter : [list of] functions
            checked and processed fitter functions

    """
    if isinstance(model, Model):
        evaluations = np.zeros(N)
    elif isinstance(model, Iterable):
        if N > 1:
            evaluations = np.zeros((N, len(model)))
        else:
            evaluations = np.zeros(len(model))
        if theta is not None:
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


def pair_tests(evaluations):
    """pairwise bootstrapping significance tests for a difference in model
    performance.
    Tests add 1/len(evaluations) to each p-value and are computed as
    two sided tests, i.e. as 2 * the smaller proportion

    Args:
        evaluations (numpy.ndarray):
            RDMs to be pooled

    Returns:
        numpy.ndarray: matrix of proportions of opposit conclusions, i.e.
        p-values for the bootstrap test
    """
    proportions = np.zeros((evaluations.shape[1], evaluations.shape[1]))
    while len(evaluations.shape) > 2:
        evaluations = np.mean(evaluations, axis=-1)
    for i_model in range(evaluations.shape[1]-1):
        for j_model in range(i_model + 1, evaluations.shape[1]):
            proportions[i_model, j_model] = np.sum(
                evaluations[:, i_model] < evaluations[:, j_model]) \
                / (evaluations.shape[0] -
                   np.sum(evaluations[:, i_model] == evaluations[:, j_model]))
            proportions[j_model, i_model] = proportions[i_model, j_model]
    proportions = np.minimum(proportions, 1 - proportions) * 2
    proportions = (len(evaluations) - 1) / len(evaluations) * proportions \
         + 1 / len(evaluations)
    np.fill_diagonal(proportions, 1)
    return proportions
