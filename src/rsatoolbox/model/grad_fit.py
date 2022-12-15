
import numpy as np
from .fitter import _normalize
from rsatoolbox.util.pooling import pool_rdm
from rsatoolbox.util.rdm_utils import _parse_nan_vectors


def fit_grad_regression(
    model, data, method='cosine', pattern_idx=None,
    pattern_descriptor=None, ridge_weight=0, sigma_k=None,
    non_negative=True
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
    theta, _ = _ls_grad(
        vectors.T, y[0],
        ridge_weight=ridge_weight,
        V=v,
        non_negative=non_negative)
    norm = np.sum(theta ** 2)
    if norm == 0:
        return theta.flatten()
    return theta.flatten() / np.sqrt(norm)


def _ls_grad(
    A, y,
    ridge_weight=0,
    V=None,
    non_negative=True,
    ls_thresh=0.8, ls_fact=2,
    verbose=False
):
    assert A.shape[0] == y.shape[0]
    assert y.ndim == 1
    x = np.ones(A.shape[1])
    if V is not None:
        V_inv = np.linalg.inv(V)
        grad = A.T @ V_inv @ (y - A @ x) - ridge_weight * x
        yA = y - A @ x
        loss = yA.T @ V @ yA
    else:
        grad = A.T @ (y - A @ x) - ridge_weight * x
        loss = np.sum((y - A @ x) ** 2)
    step_size = 1
    while np.sum(grad**2) > 1000 * np.finfo(float).eps:
        # Increase step_size while x_new improves enough
        if verbose:
            print(f"step {step_size}, l={loss}\n")
        repeat_grow = True
        repeat_shrink = True
        while repeat_grow or repeat_shrink:
            x_new = x + step_size * grad
            if non_negative:
                x_new[x_new < 0] = 0
            if V is None:
                loss_new = np.sum((y - A @ x_new) ** 2)
            else:
                yA = y - A @ x_new
                loss_new = yA.T @ V @ yA
            if loss_new < (loss - ls_thresh * np.sum(grad**2) * step_size):
                step_size = step_size * ls_fact
                repeat_shrink = False
            else:
                step_size = step_size / ls_fact
                repeat_grow = False
                # we started growing & need to go one step back
                if not repeat_shrink:
                    x_new = x + step_size * grad
                    if non_negative:
                        x_new[x_new < 0] = 0
                    if V is None:
                        loss_new = np.sum((y - A @ x_new) ** 2)
                    else:
                        yA = y - A @ x_new
                        loss_new = yA.T @ V @ yA
        x = x_new
        loss = loss_new
        if V is not None:
            grad = A.T @ V_inv @ (y - A @ x) - ridge_weight * x
        else:
            grad = A.T @ (y - A @ x) - ridge_weight * x
        if non_negative:
            grad[(x == 0) & (grad < 0)] = 0
    if V is None:
        loss = np.sum((y - A @ x) ** 2)
    else:
        loss = (y - A @ x).T @ V @ (y - A @ x)
    return x, loss
