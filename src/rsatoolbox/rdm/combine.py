"""Functions operating on a set of related RDMs objects
"""
from __future__ import annotations
from copy import deepcopy
from typing import TYPE_CHECKING, List, Optional, Tuple, Dict
import numpy as np
from numpy import sqrt, nan, inf, ndarray
from scipy.spatial.distance import squareform
import rsatoolbox.rdm.rdms
if TYPE_CHECKING:
    from rsatoolbox.rdm.rdms import RDMs


def _merged_rdm_descriptors(list_of_rdms: List[RDMs]) -> Tuple[Dict, Dict]:
    """Merge descriptors and rdm_descriptors for multiple RDMs objects

    Some descriptors have to be demoted to rdm_descriptors if they vary.
    Used by
        rsatoolbox.rdm.combine.from_partials
        rsatoolbox.rdm.rdms.concat

    Args:
        list_of_rdms (List[RDMs]): List of RDMs objects

    Returns:
        Tuple[Dict, Dict]: descriptors, rdms_descriptors
    """

    n_rdms = sum([rdms.n_rdm for rdms in list_of_rdms])
    rdm_desc_names = []
    descriptors = deepcopy(list_of_rdms[0].descriptors)
    desc_diff_names = []
    for rdms in list_of_rdms[1:]:
        rdm_desc_names += list(rdms.rdm_descriptors.keys())
        delete = []
        for k, v in descriptors.items():
            if k not in rdms.descriptors.keys():
                desc_diff_names.append(k)
                delete.append(k)
            elif not np.all(rdms.descriptors[k] == v):
                desc_diff_names.append(k)
                delete.append(k)
        for k in delete:
            descriptors.pop(k)
        for k, v in rdms.descriptors.items():
            if k not in descriptors.keys() and k not in desc_diff_names:
                desc_diff_names.append(k)

    rdm_desc_names = set(rdm_desc_names + list(desc_diff_names))
    rdm_descriptors = dict([(n, [None]*n_rdms) for n in rdm_desc_names])
    rdm_id = 0
    for rdms in list_of_rdms:
        for rdm_local_id, _ in enumerate(rdms.dissimilarities):
            for name in rdm_descriptors.keys():
                if name == 'index':
                    rdm_descriptors['index'][rdm_id] = rdm_id
                elif name in rdms.rdm_descriptors:
                    val = rdms.rdm_descriptors[name][rdm_local_id]
                    rdm_descriptors[name][rdm_id] = val
                elif name in rdms.descriptors:
                    rdm_descriptors[name][rdm_id] = rdms.descriptors[name]
                else:
                    rdm_descriptors[name] = None
            rdm_id += 1
    return descriptors, rdm_descriptors


def from_partials(
        list_of_rdms: List[RDMs],
        all_patterns: Optional[List[str]] = None,
        descriptor: str = 'conds') -> RDMs:
    """Make larger RDMs with missing values where needed

    Any object-level descriptors will be turned into rdm_descriptors
    if they do not match across objects.

    Args:
        list_of_rdms (list): List of RDMs objects
        all_patterns (list, optional): The full list of pattern
            descriptors. Defaults to None, in which case the full
            list is the union of all input rdms' values for the
            pattern descriptor chosen.
        descriptor (str, optional): The pattern descriptor on the basis
            of which to expand. Defaults to 'conds'.

    Returns:
        RDMs: Object containing all input rdms on the larger scale,
            with missing values where required
    """

    def pdescs(rdms, descriptor):
        return list(rdms.pattern_descriptors.get(descriptor, []))
    if all_patterns is None:
        all_patterns = []
        for rdms in list_of_rdms:
            all_patterns += pdescs(rdms, descriptor)
        all_patterns = list(dict.fromkeys(all_patterns).keys())

    n_rdms = sum([rdms.n_rdm for rdms in list_of_rdms])
    n_patterns = len(all_patterns)

    descriptors, rdm_descriptors = _merged_rdm_descriptors(list_of_rdms)

    measure = None
    vector_len = int(n_patterns * (n_patterns-1) / 2)
    vectors = np.full((n_rdms, vector_len), np.nan)
    rdm_id = 0
    for rdms in list_of_rdms:
        measure = rdms.dissimilarity_measure
        pidx = [all_patterns.index(i) for i in pdescs(rdms, descriptor)]
        for _, utv in enumerate(rdms.dissimilarities):
            rdm = np.full((len(all_patterns), len(all_patterns)), np.nan)
            rdm[np.ix_(pidx, pidx)] = squareform(utv, checks=False)
            vectors[rdm_id, :] = squareform(rdm, checks=False)
            rdm_id += 1
    return rsatoolbox.rdm.RDMs(
        dissimilarities=vectors,
        dissimilarity_measure=measure,
        descriptors=descriptors,
        rdm_descriptors=rdm_descriptors,
        pattern_descriptors=dict([(descriptor, all_patterns)])
    )


def rescale(rdms, method: str = 'evidence', threshold=1e-8):
    """Bring RDMs closer together

    Iteratively scales RDMs based on pairs in-common.
    Also adds an RDM descriptor with the weights used.

    Args:
        method (str, optional): One of 'evidence', 'setsize' or
            'simple'. Defaults to 'evidence'.
        threshold (float): Stop iterating when the sum of squares
            difference between iterations is smaller than this value.
            A smaller value means more iterations, but the algorithm
            may not always converge.

    Returns:
        RDMs: RDMs object with the aligned RDMs
    """
    aligned, weights = _rescale(rdms.dissimilarities, method, threshold)
    rdm_descriptors = deepcopy(rdms.rdm_descriptors)
    if weights is not None:
        rdm_descriptors['rescalingWeights'] = weights
    return rsatoolbox.rdm.rdms.RDMs(
        dissimilarities=aligned,
        dissimilarity_measure=rdms.dissimilarity_measure,
        descriptors=deepcopy(rdms.descriptors),
        rdm_descriptors=rdm_descriptors,
        pattern_descriptors=deepcopy(rdms.pattern_descriptors)
    )


def _mean(vectors: ndarray, weights: Optional[ndarray] = None) -> ndarray:
    """Weighted mean of RDM vectors, ignores nans

    See :meth:`rsatoolbox.rdm.rdms.RDMs.mean`

    Args:
        vectors (ndarray): dissimilarity vectors of shape (nrdms, nconds)
        weights (ndarray, optional): Same shape as vectors.

    Returns:
        ndarray: Average vector of shape (nconds,)
    """
    if weights is None:
        weights = np.ones(vectors.shape)
        weights[np.isnan(vectors)] = np.nan
    weighted_sum = np.nansum(vectors * weights, axis=0)
    return weighted_sum / np.nansum(weights, axis=0)


def _ss(vectors: ndarray) -> ndarray:
    """Sum of squares on the last dimension

    Args:
        vectors (ndarray): 1- or 2-dimensional data

    Returns:
        ndarray: the sum of squares, with an extra empty dimension
    """
    summed_squares = np.nansum(vectors ** 2, axis=vectors.ndim-1)
    return np.expand_dims(summed_squares, axis=vectors.ndim-1)


def _scale(vectors: ndarray) -> ndarray:
    """Divide by the root sum of squares

    Args:
        vectors (ndarray): 1- or 2-dimensional data

    Returns:
        ndarray: input scaled
    """
    return vectors / sqrt(_ss(vectors))


def _rescale(dissim: ndarray, method: str, threshold=1e-8) -> Tuple[ndarray, ndarray]:
    """Rescale RDM vectors

    See :meth:`rsatoolbox.rdm.combine.rescale`

    Args:
        dissim (ndarray): dissimilarity vectors, shape = (rdms, conds)
        method (str): one of 'evidence', 'setsize' or 'simple'.

    Returns:
        (ndarray, ndarray): Tuple of the aligned dissimilarity vectors
            and the weights used
    """
    n_rdms, n_conds = dissim.shape
    if method == 'evidence':
        weights = (dissim ** 2).clip(0.2 ** 2)
    elif method == 'setsize':
        setsize = np.isfinite(dissim).sum(axis=1)
        weights = np.tile(1 / setsize, [n_conds, 1]).T
    else:
        weights = np.ones(dissim.shape)
    weights[np.isnan(dissim)] = np.nan

    current_estimate = _scale(_mean(dissim))
    prev_estimate = np.full([n_conds, ], -inf)
    while _ss(current_estimate - prev_estimate) > threshold:
        prev_estimate = current_estimate.copy()
        tiled_estimate = np.tile(current_estimate, [n_rdms, 1])
        tiled_estimate[np.isnan(dissim)] = nan
        aligned = _scale(dissim) * sqrt(_ss(tiled_estimate))
        current_estimate = _scale(_mean(aligned, weights))

    return aligned, weights
