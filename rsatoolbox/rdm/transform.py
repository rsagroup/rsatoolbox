#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" transforms, which can be applied to RDMs
"""

from copy import deepcopy
import numpy as np
from scipy.stats import rankdata
from .rdms import RDMs


def rank_transform(rdms, method='average'):
    """ applies a rank_transform and generates a new RDMs object
    This assigns a rank to each dissimilarity estimate in the RDM,
    deals with rank ties and saves ranks as new dissimilarity estimates.
    As an effect, all non-diagonal entries of the RDM will
    range from 1 to (n_dim²-n_dim)/2, if the RDM has the dimensions
    n_dim x n_dim.

    Args:
        rdms(RDMs): RDMs object
        method(String):
            controls how ranks are assigned to equal values
            options are: ‘average’, ‘min’, ‘max’, ‘dense’, ‘ordinal’

    Returns:
        rdms_new(RDMs): RDMs object with rank transformed dissimilarities

    """
    dissimilarities = rdms.get_vectors()
    dissimilarities = np.array([rankdata(dissimilarities[i], method=method)
                                for i in range(rdms.n_rdm)])
    measure = rdms.dissimilarity_measure
    if not measure[-7:] == '(ranks)':
        measure = measure + ' (ranks)'
    rdms_new = RDMs(dissimilarities,
                    dissimilarity_measure=measure,
                    descriptors=deepcopy(rdms.descriptors),
                    rdm_descriptors=deepcopy(rdms.rdm_descriptors),
                    pattern_descriptors=deepcopy(rdms.pattern_descriptors))
    return rdms_new


def sqrt_transform(rdms):
    """ applies a square root transform and generates a new RDMs object
    This sets values blow 0 to 0 and takes a square root of each entry.
    It also adds a sqrt to the dissimilarity_measure entry.

    Args:
        rdms(RDMs): RDMs object

    Returns:
        rdms_new(RDMs): RDMs object with sqrt transformed dissimilarities

    """
    dissimilarities = rdms.get_vectors()
    dissimilarities[dissimilarities < 0] = 0
    dissimilarities = np.sqrt(dissimilarities)
    if rdms.dissimilarity_measure == 'squared euclidean':
        dissimilarity_measure = 'euclidean'
    elif rdms.dissimilarity_measure == 'squared mahalanobis':
        dissimilarity_measure = 'mahalanobis'
    else:
        dissimilarity_measure = 'sqrt of' + rdms.dissimilarity_measure
    rdms_new = RDMs(dissimilarities,
                    dissimilarity_measure=dissimilarity_measure,
                    descriptors=deepcopy(rdms.descriptors),
                    rdm_descriptors=deepcopy(rdms.rdm_descriptors),
                    pattern_descriptors=deepcopy(rdms.pattern_descriptors))
    return rdms_new


def positive_transform(rdms):
    """ sets all negative entries in an RDM to zero and returns a new RDMs

    Args:
        rdms(RDMs): RDMs object

    Returns:
        rdms_new(RDMs): RDMs object with sqrt transformed dissimilarities

    """
    dissimilarities = rdms.get_vectors()
    dissimilarities[dissimilarities < 0] = 0
    rdms_new = RDMs(dissimilarities,
                    dissimilarity_measure=rdms.dissimilarity_measure,
                    descriptors=deepcopy(rdms.descriptors),
                    rdm_descriptors=deepcopy(rdms.rdm_descriptors),
                    pattern_descriptors=deepcopy(rdms.pattern_descriptors))
    return rdms_new


def transform(rdms, fun):
    """ applies an arbitray function ``fun`` to the dissimilarities and
    returns a new RDMs object.

    Args:
        rdms(RDMs): RDMs object

    Returns:
        rdms_new(RDMs): RDMs object with sqrt transformed dissimilarities

    """
    dissimilarities = rdms.get_vectors()
    dissimilarities = fun(dissimilarities)
    rdms_new = RDMs(dissimilarities,
                    dissimilarity_measure='transformed ' + rdms.dissimilarity_measure,
                    descriptors=deepcopy(rdms.descriptors),
                    rdm_descriptors=deepcopy(rdms.rdm_descriptors),
                    pattern_descriptors=deepcopy(rdms.pattern_descriptors))
    return rdms_new
