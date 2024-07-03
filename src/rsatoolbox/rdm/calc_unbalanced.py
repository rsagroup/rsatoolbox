#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calculation of RDMs from unbalanced datasets, i.e. datasets with different
channels or numbers of measurements per dissimilarity

@author: heiko
"""
from __future__ import annotations
from typing import TYPE_CHECKING, Tuple, Union, List
from collections.abc import Iterable
from copy import deepcopy
import warnings
import numpy as np
from rsatoolbox.rdm.rdms import RDMs
from rsatoolbox.rdm.rdms import concat
from rsatoolbox.util.data_utils import get_unique_inverse
from rsatoolbox.util.matrix import row_col_indicator_rdm
from rsatoolbox.util.build_rdm import _build_rdms
from rsatoolbox.cengine.similarity import calc_one, calc
if TYPE_CHECKING:
    from rsatoolbox.data.base import DatasetBase
    from numpy.typing import NDArray
    SingleOrMultiDataset = Union[DatasetBase, List[DatasetBase]]


def calc_rdm_unbalanced(dataset: SingleOrMultiDataset, method='euclidean',
                        descriptor=None, noise=None, cv_descriptor=None,
                        prior_lambda=1, prior_weight=0.1,
                        weighting='number', enforce_same=False) -> RDMs:
    """
    calculate a RDM from an input dataset for unbalanced datasets.

    Args:
        dataset (rsatoolbox.data.dataset.DatasetBase):
            The dataset the RDM is computed from
        method (String):
            a description of the dissimilarity measure (e.g. 'Euclidean')
        descriptor (String):
            obs_descriptor used to define the rows/columns of the RDM
        noise (numpy.ndarray):
            dataset.n_channel x dataset.n_channel
            precision matrix used to calculate the RDM
            used only for Mahalanobis and Crossnobis estimators
            defaults to an identity matrix, i.e. euclidean distance

    Returns:
        rsatoolbox.rdm.rdms.RDMs: RDMs object with the one RDM

    """
    if isinstance(dataset, Iterable):
        rdms = []
        for i_dat, dat in enumerate(dataset):
            if noise is None:
                rdms.append(calc_rdm_unbalanced(
                    dat, method=method, descriptor=descriptor,
                    cv_descriptor=cv_descriptor,
                    prior_lambda=prior_lambda, prior_weight=prior_weight,
                    weighting=weighting, enforce_same=enforce_same))
            elif isinstance(noise, np.ndarray) and noise.ndim == 2:
                rdms.append(calc_rdm_unbalanced(
                    dat, method=method,
                    descriptor=descriptor,
                    noise=noise,
                    cv_descriptor=cv_descriptor,
                    prior_lambda=prior_lambda, prior_weight=prior_weight,
                    weighting=weighting, enforce_same=enforce_same))
            elif isinstance(noise, Iterable):
                rdms.append(calc_rdm_unbalanced(
                    dat, method=method,
                    descriptor=descriptor,
                    noise=noise[i_dat],
                    cv_descriptor=cv_descriptor,
                    prior_lambda=prior_lambda, prior_weight=prior_weight,
                    weighting=weighting, enforce_same=enforce_same))
        rdm = concat(rdms)
    else:
        if descriptor is None:
            dataset = deepcopy(dataset)
            dataset.obs_descriptors['index'] = np.arange(dataset.n_obs)
            descriptor = 'index'
        if method == 'crossnobis' or method == 'poisson_cv':
            if cv_descriptor is None:
                if 'index' not in dataset.obs_descriptors.keys():
                    dataset.obs_descriptors['index'] = np.arange(dataset.n_obs)
                cv_descriptor = 'index'
                warnings.warn('cv_descriptor not set, using index for now.'
                              + 'This will only remove self-similarities.'
                              + 'Effectively this assumes independent trials')
        unique_cond, cond_indices = get_unique_inverse(
            dataset.obs_descriptors[descriptor])
        # unique_cond = set(dataset.obs_descriptors[descriptor])
        if cv_descriptor is None:
            cv_desc_int = np.arange(dataset.n_obs, dtype=np.int64)
            crossval = 0
        else:
            _, indices = np.unique(
                dataset.obs_descriptors[cv_descriptor],
                return_inverse=True
            )
            cv_desc_int = indices.astype(np.int64)
            crossval = 1
        if method == 'euclidean':
            method_idx = 1
        elif method == 'correlation':
            method_idx = 2
        elif method in ['mahalanobis', 'crossnobis']:
            method_idx = 3
        elif method in ['poisson', 'poisson_cv']:
            method_idx = 4
        else:
            raise ValueError(f'Unknown method: {method}')
        if weighting == 'equal':
            weight_idx = 0
        else:
            weight_idx = 1
        cond_indices_int = cond_indices.astype(np.int64)
        rdm = calc(
            ensure_double(dataset.measurements),
            cond_indices_int,
            cv_desc_int, len(unique_cond),
            method_idx, noise,
            prior_lambda, prior_weight,
            weight_idx, crossval)
        self_sim = rdm[:len(unique_cond)]
        rdm = rdm[len(unique_cond):]
        row_idx, col_idx = row_col_indicator_rdm(len(unique_cond))
        rdm = np.array(rdm)
        self_sim = np.array(self_sim)
        rdm = row_idx @ self_sim + col_idx @ self_sim - 2 * rdm
        rdm = _build_rdms(rdm, dataset, method, descriptor, unique_cond,
                          cv_desc_int, noise)
    return rdm


def calc_one_similarity(data_i: DatasetBase, data_j: DatasetBase,
                        cv_desc_i: NDArray, cv_desc_j: NDArray,
                        method='euclidean',
                        noise=None, weighting='number',
                        prior_lambda=1, prior_weight=0.1
                        ) -> Tuple[NDArray, NDArray]:
    """
    finds all pairs of vectors to be compared and calculates one distance

    Args:
        data_i (rsatoolbox.data.DatasetBase):
            dataset for condition i
        data_j (rsatoolbox.data.DatasetBase):
            dataset for condition j
        cv_desc_i(numpy.ndarray):
            crossvalidation descriptor for condition i
        cv_desc_j(numpy.ndarray):
            crossvalidation descriptor for condition j
        method (string):
            which dissimilarity to compute
        noise : numpy.ndarray (n_channels x n_channels), optional
            the covariance or precision matrix over channels
            necessary for calculation of mahalanobis distances

    Returns:
        (np.ndarray, np.ndarray) : (value, weight)
            value is the dissimilarity
            weight is the weight of the samples

    """
    if method == 'euclidean':
        method_idx = 1
    elif method == 'correlation':
        method_idx = 2
    elif method in ['mahalanobis', 'crossnobis']:
        method_idx = 3
    elif method in ['poisson', 'poisson_cv']:
        method_idx = 4
    else:
        raise ValueError(f'Unknown method: {method}')
    if weighting == 'equal':
        weight_idx = 0
    else:
        weight_idx = 1
    return calc_one(
        ensure_double(data_i.measurements),
        ensure_double(data_j.measurements),
        cv_desc_i, cv_desc_j,
        data_i.n_obs, data_j.n_obs,
        method_idx, noise=noise,
        prior_lambda=prior_lambda, prior_weight=prior_weight,
        weighting=weight_idx)


def ensure_double(a: NDArray) -> NDArray[np.float64]:
    """If required, will convert the array datatype to Float64

    This ensures compatibility with the underlying c type "double".
    If the array is already compatible, it will pass through.
    If it is an integer, a converted copy will be made.

    Args:
        a (NDArray): Numeric numpy array

    Returns:
        NDArray[np.float64]: The float64 version of the array
    """
    return a.astype(np.float64)
