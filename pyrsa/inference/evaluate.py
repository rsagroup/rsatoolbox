#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
inference module: evaluate models
@author: heiko
"""

import numpy as np
from pyrsa.rdm import compare


def eval_fixed(model, data, method='cosine'):
    """evaluates a model on data, without any bootstrapping or
    cross-validation

    Args:
        model(pyrsa.model.Model): Model to be evaluated

        data(pyrsa.rdm.RDMs): data to evaluate on

        method(string): comparison method to use

    Returns:
        float: evaluation

    """
    rdm_pred = model.predict()
    return compare(rdm_pred, data, method)


def eval_bootstrap_rdm(model, data, method, N=1000,
                       rdm_descriptor=None):
    """evaluates a model on data
    performs 

    Args:
        model(pyrsa.model.Model): Model to be evaluated

        data(pyrsa.rdm.RDMs): data to evaluate on

        method(string): comparison method to use

        N(int): number of samples

        rdm_descriptor(string): rdm_descriptor to group rdms for bootstrap

    Returns:
        float: evaluation

    """
