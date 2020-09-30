"""Private implementations of align() and mean()
"""
import numpy as np


def _mean(vectors, weights=None):
    """Weighted mean of RDM vectors

    See :meth:`pyrsa.rdm.rdms.RDMs.mean`

    Args:
        vectors (ndarray): dissimilarity vectors of shape (nrdms, nconds)
        weights (ndarray, optional): Same shape as vectors.
            Defaults to None.

    Returns:
        ndarray: Average vector of shape (nconds,)
    """
    if weights is None:
        weights = np.ones(vectors.shape)
        weights[np.isnan(vectors)] = np.nan

    weighted_sum = np.nansum(vectors * weights, axis=0)
    return weighted_sum / np.nansum(weights, axis=0)
