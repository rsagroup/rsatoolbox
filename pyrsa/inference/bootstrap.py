#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""inference module: bootstrapping
"""

import numpy as np

def bootstrap_sample(rdms, descriptors=None, pattern_descriptors=None):
    """draws a bootstrap_sample from the data
    
    This function generates a bootstrap sample of RDMs resampled over 
    measurements and patterns. By default every pattern and RDM sample is 
    treated independently. If desired descriptor names can be passed in 
    descriptors and in pattern_descriptors to group rdms instead.
    
    Args:
        rdms(pyrsa.rdm.RDMs): Data to be used
        descriptors(list of string): 
            descriptors to group the samples by. For each unique value of 
            the descriptors each sample will either contain all RDMs with 
            this combination or none
        pattern_descriptors(list of string):
            descriptors to group the patterns by. Each group of patterns will
            be in or out of the sample as a whole
    Output:
        rdm_sample (pyrsa.rdm.RDMs):
            subsampled dataset with equal number of groups in both patterns
            and measurements of the rdms
    """
    
    return rdms
    