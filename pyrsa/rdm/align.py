"""Private implementations of align() and mean()
"""
import numpy as np


def _mean(vectors, weights):
    """Weighted mean of RDM vectors, ignores nans

    See :meth:`pyrsa.rdm.rdms.RDMs.mean`

    Args:
        vectors (ndarray): dissimilarity vectors of shape (nrdms, nconds)
        weights (ndarray, optional): Same shape as vectors.

    Returns:
        ndarray: Average vector of shape (nconds,)
    """
    weighted_sum = np.nansum(vectors * weights, axis=0)
    return weighted_sum / np.nansum(weights, axis=0)


def _align(dissim, method):
    if method == 'evidence':
        weights = (dissim ** 2).clip(0.2)
    elif method == 'setsize':
        setsize = np.isfinite(dissim).sum(axis=1)
        weights = np.tile(1 / setsize, [dissim.shape[1], 1]).T
    else:
        weights = np.ones(dissim.shape)
    weights[np.isnan(dissim)] = np.nan

    return dissim, weights
