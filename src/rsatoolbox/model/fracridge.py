#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Additional fitting method: fractional ridge regression
This is heavily based on the frrsa methods
by Phillip Kaniuth & Martin Hebart
https://github.com/ViCCo-Group/frrsa
https://doi.org/10.1016/j.neuroimage.2022.119294

These in turn are based on the fractional ridge regression implementation
by Ariel Rokem and Kendrick Kay
https://github.com/nrdg/fracridge
https://doi.org/10.1093/gigascience/giaa133

Changes by Heiko:
1) use LAPACK eigenvalue problem solver instead of SVD
2) No more jit, as it gave very small improvements at best
"""
import warnings
import numpy as np
from numpy import interp
import scipy
from .fitter import _normalize
from rsatoolbox.util.pooling import pool_rdm
from rsatoolbox.util.rdm_utils import _parse_nan_vectors

# Module-wide constants
BIG_BIAS = 10e3
SMALL_BIAS = 10e-3
BIAS_STEP = 0.2


def fit_frac_regression(
    model, data, method='cosine', pattern_idx=None,
    pattern_descriptor=None, sigma_k=None
):
    if not (pattern_idx is None or pattern_descriptor is None):
        pred = model.rdm_obj.subsample_pattern(pattern_descriptor, pattern_idx)
    else:
        pred = model.rdm_obj
    vectors = pred.get_vectors()
    data_mean = pool_rdm(data, method=method)
    y = data_mean.get_vectors()
    vectors, y, non_nan_mask = _parse_nan_vectors(vectors, y)
    vectors, v, y = _normalize(
        vectors, y, method,
        pred.n_cond, non_nan_mask, sigma_k)
    _fracridge(vectors, y)


def _fracridge(
    X, y,
    X_test=None,
    fracs=None,
    tol=1e-10,
    betas_wanted=True,
    pred_wanted=True
):
    """
    Approximates alpha parameters to match desired fractions of OLS length.
    Parameters
    ----------
    X : ndarray, shape (n, p)
        Design matrix for regression, with n number of
        observations and p number of model parameters.
    y : ndarray, shape (n, b)
        Data, with n number of observations and b number of targets.
    fracs : float or 1d array, optional
        The desired fractions of the parameter vector length, relative to
        OLS solution. If 1d array, the shape is (f,). This input is required
        to be sorted. Otherwise, raises ValueError.
        Default: np.arange(.1, 1.1, .1).
    Returns
    -------
    coef : ndarray, shape (p, f, b)
        The full estimated parameters across units of measurement for every
        desired fraction.
    alphas : ndarray, shape (f, b)
        The alpha coefficients associated with each solution
    """
    if fracs is None:
        fracs = np.arange(.1, 1.1, .1)

    if hasattr(fracs, "__len__"):
        if np.any(np.diff(fracs) < 0):
            raise ValueError("The `frac` inputs to the `fracridge` function ",
                             f"must be sorted. You provided: {fracs}")
    else:
        fracs = [fracs]
    fracs = np.array(fracs)

    nn, pp = X.shape
    if len(y.shape) == 1:
        y = y[:, np.newaxis]

    bb = y.shape[-1]
    ff = fracs.shape[0]

    # Calculate the rotation of the data
    # selt, v_t, ols_coef = _do_svd(X, y, jit=jit)
    w = scipy.linalg.lapack.dsyevd(X.T @ X)
    selt = np.sqrt(w[0])
    v_t = w[1]
    if y.shape[-1] >= X.shape[0]:
        ynew = (np.diag(1./selt) @ v_t @ X.T) @ y
    else:
        ynew = np.diag(1./selt) @ v_t @ (X.T @ y)
    ols_coef = (ynew.T / selt).T

    # Set solutions for small eigenvalues to 0 for all targets:
    isbad = selt < tol
    if np.any(isbad):
        warnings.warn("Some eigenvalues are being treated as 0")

    ols_coef[isbad, ...] = 0

    # Limits on the grid of candidate alphas used for interpolation:
    val1 = BIG_BIAS * selt[0] ** 2
    val2 = SMALL_BIAS * selt[-1] ** 2

    # Generates the grid of candidate alphas used in interpolation:
    alphagrid = np.concatenate(
        [np.array([0]),
         10 ** np.arange(np.floor(np.log10(val2)),
                         np.ceil(np.log10(val1)), BIAS_STEP)])

    # The scaling factor applied to coefficients in the rotated space is
    # lambda**2 / (lambda**2 + alpha), where lambda are the singular values
    seltsq = selt**2
    sclg = seltsq / (seltsq + alphagrid[:, None])
    sclg_sq = sclg**2

    # Prellocate the solution:
    if nn >= pp:
        first_dim = pp
    else:
        first_dim = nn

    coef = np.empty((first_dim, ff, bb))
    alphas = np.empty((ff, bb))

    # The main loop is over targets:
    for ii in range(y.shape[-1]):
        # Applies the scaling factors per alpha
        newlen = np.sqrt(sclg_sq @ ols_coef[..., ii]**2).T
        # Normalize to the length of the unregularized solution,
        # because (alphagrid[0] == 0)
        newlen = (newlen / newlen[0])
        # Perform interpolation in a log transformed space (so it behaves
        # nicely), avoiding log of 0.
        temp = interp(fracs, newlen[::-1], np.log(1 + alphagrid)[::-1])
        # Undo the log transform from the previous step
        targetalphas = np.exp(temp) - 1
        # Allocate the alphas for this target:
        alphas[:, ii] = targetalphas
        # Calculate the new scaling factor, based on the interpolated alphas:
        sc = seltsq / (seltsq + targetalphas[np.newaxis].T)
        # Use the scaling factor to calculate coefficients in the rotated
        # space:
        coef[..., ii] = (sc * ols_coef[..., ii]).T

    # After iterating over all targets, we unrotate using the unitary v
    # matrix and reshape to conform to desired output:

    if pred_wanted:
        pred = np.reshape(
            X_test @ v_t.T @ coef.reshape((first_dim, ff * bb)),
            (X_test.shape[0], ff, bb)).squeeze()
    else:
        pred = None

    if betas_wanted:
        coef = np.reshape(
            v_t.T @ coef.reshape((first_dim, ff * bb)),
            (pp, ff, bb)
        ).squeeze()
    else:
        coef = None

    return pred, coef, alphas
