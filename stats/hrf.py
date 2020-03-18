#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 10 13:07:37 2019

@author: heiko
"""

import numpy as np
import scipy.stats as stats

# SPMs HRF
def spm_hrf(t,
                   peak_delay=6,
                   under_delay=16,
                   peak_disp=1,
                   under_disp=1,
                   p_u_ratio = 6,
                   normalize_to_peak=False,
                  ):
    """ SPM HRF function from sum of two gamma PDFs
    as implemented in hrf_estimation (Pedregosa, Eickenberg, Ciuciu, Thirion, & Gramfort, 2015)

    This function is designed to be partially compatible with SPMs `spm_hrf.m`
    function.
    
    See ``spm_hrf.m`` in the SPM distribution.
    
    ## ADJUSTMENTS Heiko ##
    changed normalization to allow integral = 1 normalization and make it default
    """
    if len([v for v in [peak_delay, peak_disp, under_delay, under_disp]
            if v <= 0]):
        raise ValueError("delays and dispersions must be > 0")
    # gamma.pdf only defined for t > 0
    hrf = np.zeros(t.shape, dtype=np.float)
    pos_t = t[t > 0]
    peak = stats.gamma.pdf(pos_t,
                         peak_delay / peak_disp,
                         loc=0,
                         scale=peak_disp)
    undershoot = stats.gamma.pdf(pos_t,
                               under_delay / under_disp,
                               loc=0,
                               scale=under_disp)
    hrf[t > 0] = peak - undershoot / p_u_ratio
    if not normalize_to_peak:
        return hrf / (1-1/p_u_ratio)
    else:
        return hrf / np.max(hrf)