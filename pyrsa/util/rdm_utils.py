#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Collection of helper methods for rdm module

@author: baihan
"""

from typing import Union, List, Tuple, Dict

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
    return int(np.ceil(np.sqrt(x.shape[1] * 2)))


def add_pattern_index(rdms, pattern_descriptor):
    """
    adds index if pattern_descriptor is None

    Args:
        **rdms** (pyrsa.rdm.RDMs): rdms object to be parsed

    Returns:
        pattern_descriptor
        pattern_select

    """
    pattern_select = rdms.pattern_descriptors[pattern_descriptor]
    pattern_select = np.unique(pattern_select)
    return pattern_descriptor, pattern_select


def category_selector_to_names_and_idxs(rdms,
                                        category_selector: Union[str, List[int]]
                                        ) -> Tuple[List[str], Dict[str, List[int]]]:
    """


    Args:
        rdms (pyrsa.rdm.RDMs):
            A reference RDM stack.
        category_selector (str or List[int]):
            Either: a string specifying the `rdms.pattern_descriptor` which labels categories for each condition.
            Or: a list of ints specifying the category label for each condition in `rdms`.

    Returns:
        a tuple of:
            category_names (List[int]):
                The names of the unique pattern descriptors within the descriptor specified by `category_selector`,
                if it is a string. Otherwise generated to be "Category i" for integers i specified in
                `category_selector`.
            category_idxs (Dict[str, List[int]]):
                A dictionary mapping the strings in `category_names` to lists of integer indices of categories within
                the RDMs.

    @author: caiw
    """

    from pyrsa.rdm import RDMs
    rdms: RDMs

    _msg_arg_category_selector = ("Argument category_selector must be a string specifying a pattern_descriptor or "
                                  "a list of ints indicating RDM conditions.")

    # One unique name for each category
    category_names: List[str]
    # Dictionary maps category names to lists of condition indices
    condition_idxs: Dict[str, List[int]]

    if isinstance(category_selector, str):
        # Use a set to get unique category labels
        category_names = sorted(set(rdms.pattern_descriptors[category_selector]))
        condition_idxs = {
            category_name: [
                idx
                for idx, cat in enumerate(rdms.pattern_descriptors[category_selector])
                if cat == category_name
            ]
            for category_name in category_names
        }

    elif isinstance(category_selector, list) and all(isinstance(i, int) for i in category_selector):
        if len(category_selector) != rdms.n_cond:
            raise ValueError(_msg_arg_category_selector)
        condition_idxs = {
            f"Category {category_i}": [
                idx
                for idx, cat in enumerate(category_selector)
                if cat == category_i
            ]
            for category_i in category_selector
        }
        category_names = sorted(condition_idxs.keys())

    else:
        raise ValueError(_msg_arg_category_selector)

    return category_names, condition_idxs
