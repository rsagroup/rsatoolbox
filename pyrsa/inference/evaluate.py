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

    Returns:
        float: evaluation

    """
    rdm_pred = model.predict()
    return compare(rdm_pred, data, method)
