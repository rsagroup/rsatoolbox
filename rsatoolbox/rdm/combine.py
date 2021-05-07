"""Functions operating on a set of related RDMs objects
"""
from __future__ import annotations
from collections import Counter
from copy import deepcopy
from typing import TYPE_CHECKING, List, Optional, Tuple
import numpy as np
from scipy.spatial.distance import squareform
from rsatoolbox.rdm.rdms import RDMs
from rsatoolbox.util.rdm_utils import _mean, _scale, _ss, _rescale


def from_partials(
    list_of_rdms: List[RDMs],
    all_patterns: Optional[List[str]]=None,
    descriptor: str='conds') -> RDMs:
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
    measure = None
    vector_len = int(n_patterns * (n_patterns-1) / 2)
    vectors = np.full((n_rdms, vector_len), np.nan)
    rdm_id = 0
    for rdms in list_of_rdms:
        measure = rdms.dissimilarity_measure
        pidx = [all_patterns.index(i) for i in pdescs(rdms, descriptor)]
        for rdm_local_id, utv in enumerate(rdms.dissimilarities):
            rdm = np.full((len(all_patterns), len(all_patterns)), np.nan)
            rdm[np.ix_(pidx, pidx)] = squareform(utv, checks=False)
            vectors[rdm_id, :] = squareform(rdm, checks=False)
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
    return RDMs(
        dissimilarities=vectors,
        dissimilarity_measure=measure,
        descriptors=descriptors,
        rdm_descriptors=rdm_descriptors,
        pattern_descriptors=dict([(descriptor, all_patterns)])
    )


def rescale(rdms, method: str='evidence'):
    """Bring RDMs closer together

    Iteratively scales RDMs based on pairs in-common.
    Also adds an RDM descriptor with the weights used.

    Args:
        method (str, optional): One of 'evidence', 'setsize' or
            'simple'. Defaults to 'evidence'.

    Returns:
        RDMs: RDMs object with the aligned RDMs
    """
    aligned, weights = _rescale(rdms.dissimilarities, method)
    rdm_descriptors = deepcopy(rdms.rdm_descriptors)
    if weights is not None:
        rdm_descriptors['rescalingWeights'] = weights
    return RDMs(
        dissimilarities=aligned,
        dissimilarity_measure=rdms.dissimilarity_measure,
        descriptors=deepcopy(rdms.descriptors),
        rdm_descriptors=rdm_descriptors,
        pattern_descriptors=deepcopy(rdms.pattern_descriptors)
    )
