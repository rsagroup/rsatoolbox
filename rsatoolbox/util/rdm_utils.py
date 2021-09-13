#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Collection of helper methods for rdm module

@author: baihan
"""

from typing import Union, List, Dict

import numpy as np
from scipy.spatial.distance import squareform


def batch_to_vectors(x):
    """converts a *stack* of RDMs in vector or matrix form into vector form

    Args:
        x: stack of RDMs

    Returns:
        tuple: **v** (np.ndarray): 2D, vector form of the stack of RDMs

        **n_rdm** (int): number of rdms

        **n_cond** (int): number of conditions
    """
    if x.ndim == 2:
        v = x
        n_rdm = x.shape[0]
        n_cond = _get_n_from_reduced_vectors(x)
    elif x.ndim == 3:
        m = x
        n_rdm = x.shape[0]
        n_cond = x.shape[1]
        v = np.ndarray((n_rdm, int(n_cond * (n_cond - 1) / 2)))
        for idx in np.arange(n_rdm):
            v[idx, :] = squareform(m[idx, :, :], checks=False)
    elif x.ndim == 1:
        v = np.array([x])
        n_rdm = 1
        n_cond = _get_n_from_reduced_vectors(v)
    return v, n_rdm, n_cond


def batch_to_matrices(x):
    """converts a *stack* of RDMs in vector or matrix form into matrix form

    Args:
        **x**: stack of RDMs

    Returns:
        tuple: **v** (np.ndarray): 3D, matrix form of the stack of RDMs

        **n_rdm** (int): number of rdms

        **n_cond** (int): number of conditions
    """
    if x.ndim == 2:
        v = x
        n_rdm = x.shape[0]
        n_cond = _get_n_from_reduced_vectors(x)
        m = np.ndarray((n_rdm, n_cond, n_cond))
        for idx in np.arange(n_rdm):
            m[idx, :, :] = squareform(v[idx, :])
    elif x.ndim == 3:
        m = x
        n_rdm = x.shape[0]
        n_cond = x.shape[1]
    return m, n_rdm, n_cond


def _get_n_from_reduced_vectors(x):
    """
    calculates the size of the RDM from the vector representation

    Args:
        **x**(np.ndarray): stack of RDM vectors (2D)

    Returns:
        int: n: size of the RDM

    """
    return max(int(np.ceil(np.sqrt(x.shape[1] * 2))), 1)


def _get_n_from_length(n):
    """
    calculates the size of the RDM from the vector length

    Args:
        **x**(np.ndarray): stack of RDM vectors (2D)

    Returns:
        int: n: size of the RDM

    """
    return int(np.ceil(np.sqrt(n * 2)))


def add_pattern_index(rdms, pattern_descriptor):
    """
    adds index if pattern_descriptor is None

    Args:
        **rdms** (rsatoolbox.rdm.RDMs): rdms object to be parsed

    Returns:
        pattern_descriptor
        pattern_select

    """
    pattern_select = rdms.pattern_descriptors[pattern_descriptor]
    pattern_select = np.unique(pattern_select)
    return pattern_descriptor, pattern_select


def _parse_input_rdms(rdm1, rdm2):
    """Gets the vector representation of input RDMs, raises an error if
    the two RDMs objects have different dimensions

    Args:
        rdm1 (rsatoolbox.rdm.RDMs):
            first set of RDMs
        rdm2 (rsatoolbox.rdm.RDMs):
            second set of RDMs

    """
    if not isinstance(rdm1, np.ndarray):
        vector1 = rdm1.get_vectors()
    else:
        if len(rdm1.shape) == 1:
            vector1 = rdm1.reshape(1, -1)
        else:
            vector1 = rdm1
    if not isinstance(rdm2, np.ndarray):
        vector2 = rdm2.get_vectors()
    else:
        if len(rdm2.shape) == 1:
            vector2 = rdm2.reshape(1, -1)
        else:
            vector2 = rdm2
    if not vector1.shape[1] == vector2.shape[1]:
        raise ValueError('rdm1 and rdm2 must be RDMs of equal shape')
    nan_idx = ~np.isnan(vector1)
    vector1_no_nan = vector1[nan_idx].reshape(vector1.shape[0], -1)
    vector2_no_nan = vector2[~np.isnan(vector2)].reshape(vector2.shape[0], -1)
    if not vector1_no_nan.shape[1] == vector2_no_nan.shape[1]:
        raise ValueError('rdm1 and rdm2 have different nan positions')
    return vector1_no_nan, vector2_no_nan, nan_idx


def _extract_triu_(X):
    """ extracts the upper triangular vector as a masked view

    Args:
        X (numpy.ndarray): 2D symmetric matrix

    Returns:
        vector version of X

    """
    mask = np.triu(np.ones_like(X, dtype=bool), k=1)
    return X[mask]


def category_condition_idxs(rdms,
                            category_selector: Union[str, List[int]]
                            ) -> Dict[str, List[int]]:
    """


    Args:
        rdms (rsatoolbox.rdm.RDMs):
            A reference RDM stack.
        category_selector (str or List[int]):
            Either: a string specifying the `rdms.pattern_descriptor` which
                    labels categories for each condition.
            Or: a list of ints specifying the category label for each condition
                in `rdms`.

    Returns:
        categories (Dict[str, List[int]]):
            A dictionary mapping the strings in `category_names` to lists of
            integer indices of categories within the RDMs.

    @author: caiw
    """

    from rsatoolbox.rdm import RDMs
    rdms: RDMs

    _msg_arg_category_selector = (
        "Argument category_selector must be a string specifying a "
        "pattern_descriptor or a list of ints indicating RDM conditions."
    )

    # Dictionary maps category names to lists of condition indices
    categories: Dict[str, List[int]]

    if isinstance(category_selector, str):
        categories = {
            category_name: [
                idx
                for idx, cat in enumerate(rdms.pattern_descriptors[
                                              category_selector])
                if cat == category_name
            ]
            # Use a set to get unique category labels
            for category_name in sorted(set(rdms.pattern_descriptors[
                                                category_selector]))
        }

    elif (isinstance(category_selector, list)
          and all(isinstance(i, int) for i in category_selector)):
        if len(category_selector) != rdms.n_cond:
            raise ValueError(_msg_arg_category_selector)
        categories = {
            f"Category {category_i}": [
                idx
                for idx, cat in enumerate(category_selector)
                if cat == category_i
            ]
            for category_i in category_selector
        }

    else:
        raise ValueError(_msg_arg_category_selector)

    return categories
