#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""inference module: bootstrapping
"""

import numpy as np

def bootstrap_sample(rdms, rdm_descriptors=None, pattern_descriptors=None):
    """draws a bootstrap_sample from the data

    This function generates a bootstrap sample of RDMs resampled over 
    measurements and patterns. By default every pattern and RDM sample is 
    treated independently. If desired descriptor names can be passed in 
    descriptors and in pattern_descriptors to group rdms instead.

    Args:
        rdms(pyrsa.rdm.rdms.RDMs): Data to be used

        rdm_descriptors(list of string): 
            descriptors to group the samples by. For each unique value of 
            the descriptors each sample will either contain all RDMs with 
            this combination or none

        pattern_descriptors(list of string):
            descriptors to group the patterns by. Each group of patterns will
            be in or out of the sample as a whole

    Returns:
        pyrsa.rdm.rdms.RDMs: rdm_sample:
        subsampled dataset with equal number of groups in both patterns
        and measurements of the rdms

    """
    if rdm_descriptors is None:
        rdm_select = np.arange(rdms.n_rdm)
        rdm_idx = np.aramge(rdms.n_rdm)
    else:
        descriptor_mat = [rdms.rdm_descriptors[i] for i in rdm_descriptors]
        descriptor_mat = np.array(descriptor_mat)
        rdm_select, rdm_idx = np.unique(descriptor_mat, return_inverse=True)
    if pattern_descriptors is None:
        pattern_select = np.arange(rdms.n_cond)
        pattern_idx = np.aramge(rdms.n_cond)
    else:
        descriptor_mat = [rdms.pattern_descriptors[i] 
                          for i in pattern_descriptors]
        descriptor_mat = np.array(descriptor_mat)
        pattern_select, pattern_idx = np.unique(descriptor_mat,
                                                return_inverse=True)
    
    return rdms
