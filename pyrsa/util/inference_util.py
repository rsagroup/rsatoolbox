#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 14:56:15 2020

@author: heiko
"""

import numpy as np
from pyrsa.model import Model
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