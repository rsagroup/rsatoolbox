#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calculation of RDMs from datasets
@author: heiko, benjamin
"""
from __future__ import annotations
from collections.abc import Iterable
from copy import deepcopy
from typing import TYPE_CHECKING, Optional, Tuple
import numpy as np
from rsatoolbox.rdm.rdms import concat
from rsatoolbox.rdm.calc_unbalanced import calc_rdm_unbalanced
from rsatoolbox.rdm.combine import from_partials
from rsatoolbox.data import average_dataset_by
from rsatoolbox.util.rdm_utils import _extract_triu_
from rsatoolbox.util.build_rdm import _build_rdms

if TYPE_CHECKING:
    from rsatoolbox.data.base import DatasetBase
    from numpy.typing import NDArray


def calc_rdm(
        dataset: DatasetBase,
        method: str = 'euclidean',
        descriptor: Optional[str] = None,
        noise: Optional[NDArray] = None,
        cv_descriptor: Optional[str] = None,
        prior_lambda: float = 1,
        prior_weight: float = 0.1,
        remove_mean: bool = False):
    """
    calculates an RDM from an input dataset

    This should usually be called with the method and the descriptor argument
    to specify the dissimilarity measure and which observations in the dataset
    belong to which condition.

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
        remove_mean (bool):
            whether the mean of each pattern shall be removed before distance calculation.
            This has no effect on poisson based and correlation distances.

    Returns:
        rsatoolbox.rdm.rdms.RDMs: RDMs object with the one RDM

    """
    if isinstance(dataset, Iterable):
        rdms = []
        for i_dat, ds_i in enumerate(dataset):
            if noise is None:
                rdms.append(calc_rdm(
                    ds_i, method=method,
                    descriptor=descriptor,
                    cv_descriptor=cv_descriptor,
                    prior_lambda=prior_lambda, prior_weight=prior_weight))
            elif isinstance(noise, np.ndarray) and noise.ndim == 2:
                rdms.append(calc_rdm(
                    ds_i, method=method,
                    descriptor=descriptor,
                    noise=noise,
                    cv_descriptor=cv_descriptor,
                    prior_lambda=prior_lambda, prior_weight=prior_weight))
            elif isinstance(noise, Iterable):
                rdms.append(calc_rdm(
                    ds_i, method=method,
                    descriptor=descriptor,
                    noise=noise[i_dat],
                    cv_descriptor=cv_descriptor,
                    prior_lambda=prior_lambda, prior_weight=prior_weight))
        if descriptor is None:
            rdm = concat(rdms)
        else:
            rdm = from_partials(rdms, descriptor=descriptor)
    else:
        if method == 'euclidean':
            rdm = calc_rdm_euclidean(dataset, descriptor, remove_mean)
        elif method == 'correlation':
            rdm = calc_rdm_correlation(dataset, descriptor)
        elif method == 'mahalanobis':
            rdm = calc_rdm_mahalanobis(dataset, descriptor, noise, remove_mean)
        elif method == 'crossnobis':
            rdm = calc_rdm_crossnobis(dataset, descriptor, noise,
                                      cv_descriptor, remove_mean)
        elif method == 'poisson':
            rdm = calc_rdm_poisson(dataset, descriptor,
                                   prior_lambda=prior_lambda,
                                   prior_weight=prior_weight)
        elif method == 'poisson_cv':
            rdm = calc_rdm_poisson_cv(dataset, descriptor,
                                      cv_descriptor=cv_descriptor,
                                      prior_lambda=prior_lambda,
                                      prior_weight=prior_weight)
        else:
            raise NotImplementedError
        if descriptor is not None:
            rdm.sort_by(**{descriptor: 'alpha'})
    return rdm


def calc_rdm_movie(
        dataset, method='euclidean', descriptor=None, noise=None,
        cv_descriptor=None, prior_lambda=1, prior_weight=0.1,
        time_descriptor='time', bins=None, unbalanced=False):
    """
    calculates an RDM movie from an input TemporalDataset

    Args:
        dataset (rsatoolbox.data.dataset.TemporalDataset):
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
        time_descriptor (String): descriptor key that points to the time
            dimension in dataset.time_descriptors. Defaults to 'time'.
        bins (array-like): list of bins, with bins[i] containing the vector
            of time-points for the i-th bin. Defaults to no binning.
        unbalanced (bool): if set to True use calc_rdm_unbalanced,
            else and by default use calc_rdm

    Returns:
        rsatoolbox.rdm.rdms.RDMs: RDMs object with RDM movie
    """

    if isinstance(dataset, Iterable):
        rdms = []
        for i_dat, ds_i in enumerate(dataset):
            if noise is None:
                rdms.append(calc_rdm_movie(
                    ds_i, method=method,
                    descriptor=descriptor))
            elif isinstance(noise, np.ndarray) and noise.ndim == 2:
                rdms.append(calc_rdm_movie(
                    ds_i, method=method,
                    descriptor=descriptor,
                    noise=noise))
            elif isinstance(noise, Iterable):
                rdms.append(calc_rdm_movie(
                    ds_i, method=method,
                    descriptor=descriptor,
                    noise=noise[i_dat]))
        rdm = concat(rdms)
    else:
        if bins is not None:
            binned_data = dataset.bin_time(time_descriptor, bins)
            splited_data = binned_data.split_time(time_descriptor)
            time = binned_data.time_descriptors[time_descriptor]
        else:
            splited_data = dataset.split_time(time_descriptor)
            time = dataset.time_descriptors[time_descriptor]

        rdms = []
        for dat in splited_data:
            dat_single = dat.time_as_observations(time_descriptor)
            if unbalanced:
                rdms.append(calc_rdm_unbalanced(
                    dat_single, method=method,
                    descriptor=descriptor, noise=noise,
                    cv_descriptor=cv_descriptor,
                    prior_lambda=prior_lambda,
                    prior_weight=prior_weight))
            else:
                rdms.append(calc_rdm(
                    dat_single, method=method,
                    descriptor=descriptor, noise=noise,
                    cv_descriptor=cv_descriptor,
                    prior_lambda=prior_lambda,
                    prior_weight=prior_weight))

        rdm = concat(rdms)
        rdm.rdm_descriptors[time_descriptor] = time
    rdm.dissimilarity_measure = method
    return rdm


def calc_rdm_euclidean(
        dataset: DatasetBase,
        descriptor: Optional[str] = None,
        remove_mean: bool = False):
    """
    Args:
        dataset (rsatoolbox.data.DatasetBase):
            The dataset the RDM is computed from
        descriptor (String):
            obs_descriptor used to define the rows/columns of the RDM
            defaults to one row/column per row in the dataset
        remove_mean (bool):
            whether the mean of each pattern shall be removed
            before calculating distances.
    Returns:
        rsatoolbox.rdm.rdms.RDMs: RDMs object with the one RDM
    """
    measurements, desc = _parse_input(dataset, descriptor, remove_mean)
    sum_sq_measurements = np.sum(measurements**2, axis=1, keepdims=True)
    rdm = sum_sq_measurements + sum_sq_measurements.T \
        - 2 * np.dot(measurements, measurements.T)
    rdm = _extract_triu_(rdm) / measurements.shape[1]
    return _build_rdms(rdm, dataset, 'squared euclidean', descriptor, desc)


def calc_rdm_correlation(dataset, descriptor=None):
    """
    calculates an RDM from an input dataset using correlation distance
    If multiple instances of the same condition are found in the dataset
    they are averaged.

    Args:
        dataset (rsatoolbox.data.DatasetBase):
            The dataset the RDM is computed from
        descriptor (String):
            obs_descriptor used to define the rows/columns of the RDM
            defaults to one row/column per row in the dataset

    Returns:
        rsatoolbox.rdm.rdms.RDMs: RDMs object with the one RDM

    """
    ma, desc = _parse_input(dataset, descriptor, remove_mean=True)
    ma /= np.sqrt(np.einsum('ij,ij->i', ma, ma))[:, None]
    rdm = 1 - np.einsum('ik,jk', ma, ma)
    return _build_rdms(rdm, dataset, 'correlation', descriptor, desc)


def calc_rdm_mahalanobis(dataset, descriptor=None, noise=None, remove_mean: bool = False):
    """
    calculates an RDM from an input dataset using mahalanobis distance
    If multiple instances of the same condition are found in the dataset
    they are averaged.

    Args:
        dataset (rsatoolbox.data.dataset.DatasetBase):
            The dataset the RDM is computed from
        descriptor (String):
            obs_descriptor used to define the rows/columns of the RDM
            defaults to one row/column per row in the dataset
        noise (numpy.ndarray):
            dataset.n_channel x dataset.n_channel
            precision matrix used to calculate the RDM
            default: identity matrix, i.e. euclidean distance
        remove_mean (bool):
            whether the mean of each pattern shall be removed
            before calculating distances.

    Returns:
        rsatoolbox.rdm.rdms.RDMs: RDMs object with the one RDM

    """
    if noise is None:
        return calc_rdm_euclidean(dataset, descriptor, remove_mean)
    measurements, desc = _parse_input(dataset, descriptor, remove_mean)
    noise = _check_noise(noise, dataset.n_channel)
    kernel = measurements @ noise @ measurements.T
    rdm = np.expand_dims(np.diag(kernel), 0) + \
        np.expand_dims(np.diag(kernel), 1) - 2 * kernel
    rdm = _extract_triu_(rdm) / measurements.shape[1]
    return _build_rdms(
        rdm,
        dataset,
        'squared mahalanobis',
        descriptor,
        desc,
        noise=noise
    )


def calc_rdm_crossnobis(dataset, descriptor, noise=None,
                        cv_descriptor=None, remove_mean: bool = False):
    """
    calculates an RDM from an input dataset using Cross-nobis distance
    This performs leave one out crossvalidation over the cv_descriptor.

    As the minimum input provide a dataset and a descriptor-name to
    define the rows & columns of the RDM.
    You may pass a noise precision. If you don't an identity is assumed.
    Also a cv_descriptor can be passed to define the crossvalidation folds.
    It is recommended to do this, to assure correct calculations. If you do
    not, this function infers a split in order of the dataset, which is
    guaranteed to fail if there are any unbalances.

    This function also accepts a list of noise precision matricies.
    It is then assumed that this is the precision of the mean from
    the corresponding crossvalidation fold, i.e. if multiple measurements
    enter a fold, please compute the resulting noise precision in advance!

    To assert equal ordering in the folds the dataset is initially sorted
    according to the descriptor used to define the patterns.

    Args:
        dataset (rsatoolbox.data.dataset.DatasetBase):
            The dataset the RDM is computed from
        descriptor (String):
            obs_descriptor used to define the rows/columns of the RDM
            defaults to one row/column per row in the dataset
        noise (numpy.ndarray):
            dataset.n_channel x dataset.n_channel
            precision matrix used to calculate the RDM
            default: identity matrix, i.e. euclidean distance
        cv_descriptor (String):
            obs_descriptor which determines the cross-validation folds
        remove_mean (bool):
            whether the mean of each pattern shall be removed
            before calculating distances.

    Returns:
        rsatoolbox.rdm.rdms.RDMs: RDMs object with the one RDM

    """
    noise = _check_noise(noise, dataset.n_channel)
    if noise is None:
        noise = np.eye(dataset.n_channel)
    if descriptor is None:
        raise ValueError('descriptor must be a string! Crossvalidation' +
                         'requires multiple measurements to be grouped')
    datasetCopy = deepcopy(dataset)
    if cv_descriptor is None:
        cv_desc = _gen_default_cv_descriptor(datasetCopy, descriptor)
        datasetCopy.obs_descriptors['cv_desc'] = cv_desc
        cv_descriptor = 'cv_desc'
    datasetCopy.sort_by(descriptor)
    cv_folds = np.unique(np.array(datasetCopy.obs_descriptors[cv_descriptor]))
    rdms = []
    if (noise is None) or (isinstance(noise, np.ndarray) and noise.ndim == 2):
        for i_fold, fold in enumerate(cv_folds):
            data_test = datasetCopy.subset_obs(cv_descriptor, fold)
            data_train = datasetCopy.subset_obs(
                cv_descriptor,
                np.setdiff1d(cv_folds, fold)
            )
            measurements_train, _, _ = \
                average_dataset_by(data_train, descriptor)
            measurements_test, _, _ = \
                average_dataset_by(data_test, descriptor)
            if remove_mean:
                measurements_train -= measurements_train.mean(axis=1, keepdims=True)
                measurements_test -= measurements_test.mean(axis=1, keepdims=True)
            rdm = _calc_rdm_crossnobis_single(
                measurements_train, measurements_test, noise)
            rdms.append(rdm)
    else:  # a list of noises was provided
        measurements = []
        variances = []
        for i, i_fold in enumerate(cv_folds):
            data = datasetCopy.subset_obs(cv_descriptor, i_fold)
            ma = average_dataset_by(data, descriptor)[0]
            if remove_mean:
                ma -= ma.mean(axis=1, keepdims=True)
            measurements.append(ma)
            variances.append(np.linalg.inv(noise[i]))
        for i_fold in range(len(cv_folds)):
            for j_fold in range(i_fold + 1, len(cv_folds)):
                if i_fold != j_fold:
                    rdm = _calc_rdm_crossnobis_single(
                        measurements[i_fold], measurements[j_fold],
                        np.linalg.inv(
                            (variances[i_fold] + variances[j_fold]) / 2)
                        )
                    rdms.append(rdm)
    rdms = np.array(rdms)
    rdm = np.einsum('ij->j', rdms) / rdms.shape[0]
    return _build_rdms(
        rdm,
        datasetCopy,
        'crossnobis',
        descriptor,
        noise=noise,
        cv=cv_descriptor
    )


def calc_rdm_poisson(dataset, descriptor=None, prior_lambda=1,
                     prior_weight=0.1):
    """
    calculates an RDM from an input dataset using the symmetrized
    KL-divergence assuming a poisson distribution.
    If multiple instances of the same condition are found in the dataset
    they are averaged.

    Args:
        dataset (rsatoolbox.data.DatasetBase):
            The dataset the RDM is computed from
        descriptor (String):
            obs_descriptor used to define the rows/columns of the RDM
            defaults to one row/column per row in the dataset

    Returns:
        rsatoolbox.rdm.rdms.RDMs: RDMs object with the one RDM

    """
    measurements, desc = _parse_input(dataset, descriptor)
    measurements = (measurements + prior_lambda * prior_weight) \
        / (1 + prior_weight)
    kernel = measurements @ np.log(measurements).T
    rdm = np.expand_dims(np.diag(kernel), 0) + \
        np.expand_dims(np.diag(kernel), 1) - kernel - kernel.T
    rdm = _extract_triu_(rdm) / measurements.shape[1]
    return _build_rdms(rdm, dataset, 'poisson', descriptor, desc)


def calc_rdm_poisson_cv(dataset, descriptor=None, prior_lambda=1,
                        prior_weight=0.1, cv_descriptor=None):
    """
    calculates an RDM from an input dataset using the crossvalidated
    symmetrized KL-divergence assuming a poisson distribution

    To assert equal ordering in the folds the dataset is initially sorted
    according to the descriptor used to define the patterns.

    Args:
        dataset (rsatoolbox.data.DatasetBase):
            The dataset the RDM is computed from
        descriptor (String):
            obs_descriptor used to define the rows/columns of the RDM
            defaults to one row/column per row in the dataset
        cv_descriptor (str): The descriptor that indicates the folds
            to use for crossvalidation

    Returns:
        rsatoolbox.rdm.rdms.RDMs: RDMs object with the one RDM

    """
    if descriptor is None:
        raise ValueError('descriptor must be a string! Crossvalidation' +
                         'requires multiple measurements to be grouped')
    dataset = deepcopy(dataset)
    if cv_descriptor is None:
        cv_desc = _gen_default_cv_descriptor(dataset, descriptor)
        dataset.obs_descriptors['cv_desc'] = cv_desc
        cv_descriptor = 'cv_desc'
    dataset.sort_by(descriptor)
    cv_folds = np.unique(np.array(dataset.obs_descriptors[cv_descriptor]))
    for i_fold in range(len(cv_folds)):
        fold = cv_folds[i_fold]
        data_test = dataset.subset_obs(cv_descriptor, fold)
        data_train = dataset.subset_obs(cv_descriptor,
                                        np.setdiff1d(cv_folds, fold))
        measurements_train, _, _ = average_dataset_by(data_train, descriptor)
        measurements_test, _, _ = average_dataset_by(data_test, descriptor)
        measurements_train = (measurements_train
                              + prior_lambda * prior_weight) \
            / (1 + prior_weight)
        measurements_test = (measurements_test
                             + prior_lambda * prior_weight) \
            / (1 + prior_weight)
        kernel = measurements_train @ np.log(measurements_test).T
        rdm = np.expand_dims(np.diag(kernel), 0) + \
            np.expand_dims(np.diag(kernel), 1) - kernel - kernel.T
        rdm = _extract_triu_(rdm) / measurements_train.shape[1]
    return _build_rdms(rdm, dataset, 'poisson_cv', descriptor)


def _calc_rdm_crossnobis_single(meas1, meas2, noise) -> NDArray:
    kernel = meas1 @ noise @ meas2.T
    rdm = np.expand_dims(np.diag(kernel), 0) + \
        np.expand_dims(np.diag(kernel), 1) - kernel - kernel.T
    return _extract_triu_(rdm) / meas1.shape[1]


def _gen_default_cv_descriptor(dataset, descriptor) -> np.ndarray:
    """ generates a default cv_descriptor for crossnobis
    This assumes that the first occurence each descriptor value forms the
    first group, the second occurence forms the second group, etc.
    """
    desc = np.asarray(dataset.obs_descriptors[descriptor])
    values, counts = np.unique(desc, return_counts=True)
    assert np.all(counts == counts[0]), (
        'cv_descriptor generation failed:\n'
        + 'different number of observations per pattern')
    n_repeats = counts[0]
    cv_descriptor = np.zeros_like(desc)
    for i_val in values:
        cv_descriptor[desc == i_val] = np.arange(n_repeats)
    return cv_descriptor


def _parse_input(
            dataset: DatasetBase,
            descriptor: Optional[str],
            remove_mean: bool = False
        ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    if descriptor is None:
        measurements = dataset.measurements
        desc = None
    else:
        measurements, desc, _ = average_dataset_by(dataset, descriptor)
    if remove_mean:
        measurements = measurements - measurements.mean(axis=1, keepdims=True)
    return measurements, desc


def _check_noise(noise, n_channel):
    """
    checks that a noise pattern is a matrix with correct dimension
    n_channel x n_channel

    Args:
        noise: noise input to be checked

    Returns:
        noise(np.ndarray): n_channel x n_channel noise precision matrix

    """
    if noise is None:
        pass
    elif isinstance(noise, np.ndarray) and noise.ndim == 2:
        assert np.all(noise.shape == (n_channel, n_channel))
    elif isinstance(noise, dict):
        for key in noise.keys():
            noise[key] = _check_noise(noise[key], n_channel)
    elif isinstance(noise, Iterable):
        for idx, noise_i in enumerate(noise):
            noise[idx] = _check_noise(noise_i, n_channel)
    else:
        raise ValueError('noise(s) must have shape n_channel x n_channel')
    return noise
