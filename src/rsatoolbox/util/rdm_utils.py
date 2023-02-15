#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Collection of helper methods for rdm module

@author: baihan
"""
from __future__ import annotations
from typing import Union, List, Dict, Tuple, TYPE_CHECKING
import numpy as np
from scipy.spatial.distance import squareform

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from rsatoolbox.rdm.rdms import RDMs


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


def _parse_input_rdms(rdm1: RDMs, rdm2: RDMs) -> Tuple[NDArray, NDArray, NDArray]:
    """Gets the vector representation of input RDMs, raises an error if
    the two RDMs objects have different dimensions, and remove nans

    Args:
        rdm1 (RDMs): first set of RDMs
        rdm2 (RDMs): second set of RDMs

    Returns:
        Tuple[NDArray, NDArray, NDArray]: Tuple of three:
            0) vector of dissimilarities for rdm1 without nans
            1) vector of dissimilarities for rdm2 without nans
            2) boolean mask of non-nan pairs
    """
    vector1 = rdm1.get_vectors()
    vector2 = rdm2.get_vectors()
    return _parse_nan_vectors(vector1, vector2)


def _parse_nan_vectors(
    vector1: NDArray, vector2: NDArray
) -> Tuple[NDArray, NDArray, NDArray]:
    """Remove nans from two dissimilarity vectors

    Args:
        vector1 (NDArray): first set of dissimilarity vectors
        vector2 (NDArray): second set of dissimilarity vectors

    Returns:
        Tuple[NDArray, NDArray, NDArray]: Tuple of three:
            0) vector of dissimilarities for vector1 without nans
            1) vector of dissimilarities for vector2 without nans
            2) boolean mask of non-nan pairs
    """
    if not vector1.shape[1] == vector2.shape[1]:
        raise ValueError("rdm1 and rdm2 must be RDMs of equal shape")
    not_nan_mask = ~np.isnan(vector1)
    vector1_no_nan = vector1[not_nan_mask].reshape(vector1.shape[0], -1)
    vector2_no_nan = vector2[~np.isnan(vector2)].reshape(vector2.shape[0], -1)
    if not vector1_no_nan.shape[1] == vector2_no_nan.shape[1]:
        raise ValueError("rdm1 and rdm2 have different nan positions")
    return vector1_no_nan, vector2_no_nan, not_nan_mask


def _extract_triu_(X):
    """extracts the upper triangular vector as a masked view

    Args:
        X (numpy.ndarray): 2D symmetric matrix

    Returns:
        vector version of X

    """
    mask = np.triu(np.ones_like(X, dtype=bool), k=1)
    return X[mask]


def category_condition_idxs(
    rdms, category_selector: Union[str, List[int]]
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
                for idx, cat in enumerate(rdms.pattern_descriptors[category_selector])
                if cat == category_name
            ]
            # Use a set to get unique category labels
            for category_name in sorted(
                set(rdms.pattern_descriptors[category_selector])
            )
        }

    elif isinstance(category_selector, list) and all(
        isinstance(i, int) for i in category_selector
    ):
        if len(category_selector) != rdms.n_cond:
            raise ValueError(_msg_arg_category_selector)
        categories = {
            f"Category {category_i}": [
                idx for idx, cat in enumerate(category_selector) if cat == category_i
            ]
            for category_i in category_selector
        }

    else:
        raise ValueError(_msg_arg_category_selector)

    return categories


def _mean(vectors: ndarray, weights: ndarray = None) -> ndarray:
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
    summed_squares = np.nansum(vectors**2, axis=vectors.ndim - 1)
    return np.expand_dims(summed_squares, axis=vectors.ndim - 1)


def _scale(vectors: ndarray) -> ndarray:
    """Divide by the root sum of squares

    Args:
        vectors (ndarray): 1- or 2-dimensional data

    Returns:
        ndarray: input scaled
    """
    return vectors / np.sqrt(_ss(vectors))


def _rescale(dissim: ndarray, method: str) -> Tuple[ndarray, ndarray]:
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
    if method == "evidence":
        weights = (dissim**2).clip(0.2**2)
    elif method == "setsize":
        setsize = np.isfinite(dissim).sum(axis=1)
        weights = np.tile(1 / setsize, [n_conds, 1]).T
    else:
        weights = np.ones(dissim.shape)
    weights[np.isnan(dissim)] = np.nan

    current_estimate = _scale(_mean(dissim))
    prev_estimate = np.full(
        [
            n_conds,
        ],
        -np.inf,
    )
    while _ss(current_estimate - prev_estimate) > 1e-8:
        prev_estimate = current_estimate.copy()
        tiled_estimate = np.tile(current_estimate, [n_rdms, 1])
        tiled_estimate[np.isnan(dissim)] = np.nan
        aligned = _scale(dissim) * np.sqrt(_ss(tiled_estimate))
        current_estimate = _scale(_mean(aligned, weights))

    return aligned, weights
