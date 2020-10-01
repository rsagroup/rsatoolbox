"""Private implementations of align() and mean()
"""
import numpy as np
from numpy import sqrt, nan, inf


def _mean(vectors, weights=None):
    """Weighted mean of RDM vectors, ignores nans

    See :meth:`pyrsa.rdm.rdms.RDMs.mean`

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


def _ss(vectors):
    """Sum of squares on the last dimension

    Args:
        vectors (ndarray): 1- or 2-dimensional data

    Returns:
        ndarray: the sum of squares, with an extra empty dimension
    """
    summed_squares = np.nansum(vectors ** 2, axis=vectors.ndim-1)
    return np.expand_dims(summed_squares, axis=vectors.ndim-1)


def _scale(vectors):
    """Divide by the root sum of squares

    Args:
        vectors (ndarray): 1- or 2-dimensional data

    Returns:
        ndarray: input scaled
    """
    return vectors / sqrt(_ss(vectors))


def _align(dissim, method):
    """Align RDM vectors

    See :meth:`pyrsa.rdm.rdms.RDMs.align`

    Args:
        dissim (ndarray): dissimilarity vectors, shape = (rdms, conds)
        method (str): one of 'evidence', 'setsize' or 'simple'.

    Returns:
        (ndarray, ndarray): Tuple of the aligned dissimilarity vectors
            and the weights used
    """
    n_rdms, n_conds = dissim.shape
    if method == 'evidence':
        weights = (dissim ** 2).clip(0.2)
    elif method == 'setsize':
        setsize = np.isfinite(dissim).sum(axis=1)
        weights = np.tile(1 / setsize, [n_conds, 1]).T
    else:
        weights = np.ones(dissim.shape)
    weights[np.isnan(dissim)] = np.nan

    current_estimate = _scale(_mean(dissim))
    prev_estimate = np.full([n_conds,], -inf)
    while _ss(current_estimate - prev_estimate) > 1e-8:
        prev_estimate = current_estimate.copy()
        tiled_estimate = np.tile(current_estimate, [n_rdms, 1])
        tiled_estimate[np.isnan(dissim)] = nan
        aligned = _scale(dissim) * sqrt(_ss(tiled_estimate))
        current_estimate = _scale(_mean(aligned, weights))

    return aligned, weights
