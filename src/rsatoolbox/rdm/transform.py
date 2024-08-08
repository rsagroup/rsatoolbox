#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" transforms, which can be applied to RDMs
"""
from __future__ import annotations
from copy import deepcopy
import numpy as np
import networkx as nx
from scipy.stats import rankdata
from scipy.spatial.distance import squareform
from .rdms import RDMs


def rank_transform(rdms: RDMs, method: str = 'average') -> RDMs:
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
    cfg = dict(method=method, nan_policy='omit')
    dissimilarities = np.array(
        [rankdata(dissimilarities[i], **cfg) for i in range(rdms.n_rdm)]
    )
    measure = rdms.dissimilarity_measure or ''
    if '(ranks)' not in measure:
        measure = (measure + ' (ranks)').strip()
    return RDMs(
        dissimilarities,
        dissimilarity_measure=measure,
        descriptors=deepcopy(rdms.descriptors),
        rdm_descriptors=deepcopy(rdms.rdm_descriptors),
        pattern_descriptors=deepcopy(rdms.pattern_descriptors)
    )


def sqrt_transform(rdms: RDMs) -> RDMs:
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
    if rdms.dissimilarity_measure is None:
        dissimilarity_measure = 'sqrt of unknown measure'
    elif rdms.dissimilarity_measure == 'squared euclidean':
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


def positive_transform(rdms: RDMs) -> RDMs:
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


def transform(rdms: RDMs, fun) -> RDMs:
    """ applies an arbitray function ``fun`` to the dissimilarities and
    returns a new RDMs object.

    Args:
        rdms(RDMs): RDMs object

    Returns:
        rdms_new(RDMs): RDMs object with sqrt transformed dissimilarities

    """
    dissimilarities = rdms.get_vectors()
    dissimilarities = fun(dissimilarities)
    if rdms.dissimilarity_measure is None:
        meas = 'transformed unknown measure'
    else:
        meas = 'transformed ' + rdms.dissimilarity_measure
    rdms_new = RDMs(dissimilarities,
                    dissimilarity_measure=meas,
                    descriptors=deepcopy(rdms.descriptors),
                    rdm_descriptors=deepcopy(rdms.rdm_descriptors),
                    pattern_descriptors=deepcopy(rdms.pattern_descriptors))
    return rdms_new


def minmax_transform(rdms: RDMs) -> RDMs:
    '''applies a minmax transform to the dissimilarities and returns a new
    RDMs object.

    Args:
        rdms(RDMs): RDMs object

    Returns:
        rdms_new(RDMs): RDMs object with minmax transformed dissimilarities
    '''
    dissimilarities = rdms.get_vectors()
    for i in range(rdms.n_rdm):
        d_max = dissimilarities[i].max()
        d_min = dissimilarities[i].min()
        dissimilarities[i] = (dissimilarities[i] - d_min) / (d_max - d_min)
    if rdms.dissimilarity_measure is None:
        meas = 'minmax transformed unknown measure'
    else:
        meas = 'minmax transformed ' + rdms.dissimilarity_measure
    rdms_new = RDMs(dissimilarities,
                    dissimilarity_measure=meas,
                    descriptors=deepcopy(rdms.descriptors),
                    rdm_descriptors=deepcopy(rdms.rdm_descriptors),
                    pattern_descriptors=deepcopy(rdms.pattern_descriptors))
    return rdms_new


def geotopological_transform(rdms: RDMs, low: float, up: float) -> RDMs:
    '''applies a geo-topological transform to the dissimilarities and returns
    a new RDMs object.

    Reference: Lin, B., & Kriegeskorte, N. (2023). The Topology and Geometry
    of Neural Representations. arXiv preprint arXiv:2309.11028.

    Args:
        rdms(RDMs): RDMs object
        low(float): lower quantile
        up(float): upper quantile

    Returns:
        rdms_new(RDMs): RDMs object with geotopological transformed dissimilarities
    '''
    dissimilarities = rdms.get_vectors()
    gt_min = np.quantile(dissimilarities, low)
    gt_max = np.quantile(dissimilarities, up)
    dissimilarities[dissimilarities < gt_min] = 0
    dissimilarities[(dissimilarities >= gt_min) & (dissimilarities <= gt_max)] = (
        dissimilarities[(dissimilarities >= gt_min) & (dissimilarities <= gt_max)] - gt_min
    ) / (gt_max - gt_min)
    dissimilarities[dissimilarities > gt_max] = 1
    if rdms.dissimilarity_measure is None:
        meas = 'geo-topological transformed unknown measure'
    else:
        meas = 'geo-topological transformed ' + rdms.dissimilarity_measure
    rdms_new = RDMs(dissimilarities,
                    dissimilarity_measure=meas,
                    descriptors=deepcopy(rdms.descriptors),
                    rdm_descriptors=deepcopy(rdms.rdm_descriptors),
                    pattern_descriptors=deepcopy(rdms.pattern_descriptors))
    return rdms_new


def geodesic_transform(rdms: RDMs) -> RDMs:
    '''applies a geodesic transform to the dissimilarities and returns a
    new RDMs object.

    Reference: Lin, B., & Kriegeskorte, N. (2023). The Topology and Geometry
    of Neural Representations. arXiv preprint arXiv:2309.11028.

    Args:
        rdms(RDMs): RDMs object

    Returns:
        rdms_new(RDMs): RDMs object with geodesic transformed dissimilarities
    '''
    dissimilarities = minmax_transform(rdms).get_vectors()
    for i in range(rdms.n_rdm):
        G = nx.from_numpy_array(squareform(dissimilarities[i]))
        long_edges = []
        long_edges = list(
            filter(lambda e: e[2] == 1, (e for e in G.edges.data("weight"))))
        le_ids = list(e[:2] for e in long_edges)
        G.remove_edges_from(le_ids)
        dissimilarities[i] = squareform(np.array(nx.floyd_warshall_numpy(G)))
    if rdms.dissimilarity_measure is None:
        meas = 'geodesic transformed unknown measure'
    else:
        meas = 'geodesic transformed ' + rdms.dissimilarity_measure
    rdms_new = RDMs(dissimilarities,
                    dissimilarity_measure=meas,
                    descriptors=deepcopy(rdms.descriptors),
                    rdm_descriptors=deepcopy(rdms.rdm_descriptors),
                    pattern_descriptors=deepcopy(rdms.pattern_descriptors))
    return rdms_new
