#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calculation of RDMs from datasets
@author: heiko, benjamin
"""

from collections.abc import Iterable
from copy import deepcopy
import numpy as np
from rsatoolbox.rdm.rdms import RDMs
from rsatoolbox.rdm.rdms import concat
from rsatoolbox.rdm.combine import from_partials
from rsatoolbox.data import average_dataset_by
from rsatoolbox.util.rdm_utils import _extract_triu_


def calc_rdm(dataset, method='euclidean', descriptor=None, noise=None,
             cv_descriptor=None, prior_lambda=1, prior_weight=0.1):
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

    Returns:
        rsatoolbox.rdm.rdms.RDMs: RDMs object with the one RDM

    """
    if isinstance(dataset, Iterable):
        rdms = []
        for i_dat in range(len(dataset)):
            if noise is None:
                rdms.append(calc_rdm(
                    dataset[i_dat], method=method,
                    descriptor=descriptor,
                    cv_descriptor=cv_descriptor,
                    prior_lambda=prior_lambda, prior_weight=prior_weight))
            elif isinstance(noise, np.ndarray) and noise.ndim == 2:
                rdms.append(calc_rdm(
                    dataset[i_dat], method=method,
                    descriptor=descriptor,
                    noise=noise,
                    cv_descriptor=cv_descriptor,
                    prior_lambda=prior_lambda, prior_weight=prior_weight))
            elif isinstance(noise, Iterable):
                rdms.append(calc_rdm(
                    dataset[i_dat], method=method,
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
            rdm = calc_rdm_euclid(dataset, descriptor)
        elif method == 'correlation':
            rdm = calc_rdm_correlation(dataset, descriptor)
        elif method == 'mahalanobis':
            rdm = calc_rdm_mahalanobis(dataset, descriptor, noise)
        elif method == 'crossnobis':
            rdm = calc_rdm_crossnobis(dataset, descriptor, noise,
                                      cv_descriptor)
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
            raise(NotImplementedError)
        if descriptor is not None:
            rdm.sort_by(**{descriptor: 'alpha'})
    return rdm


def calc_rdm_movie(
        dataset, method='euclidean', descriptor=None, noise=None,
        cv_descriptor=None, prior_lambda=1, prior_weight=0.1,
        time_descriptor='time', bins=None):
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
        time_descriptor (String): descriptor key that points to the time dimension in
            dataset.time_descriptors. Defaults to 'time'.
        bins (array-like): list of bins, with bins[i] containing the vector
            of time-points for the i-th bin. Defaults to no binning.

    Returns:
        rsatoolbox.rdm.rdms.RDMs: RDMs object with RDM movie
    """

    if isinstance(dataset, Iterable):
        rdms = []
        for i_dat, _ in enumerate(dataset):
            if noise is None:
                rdms.append(calc_rdm_movie(
                    dataset[i_dat], method=method,
                    descriptor=descriptor))
            elif isinstance(noise, np.ndarray) and noise.ndim == 2:
                rdms.append(calc_rdm_movie(
                    dataset[i_dat], method=method,
                    descriptor=descriptor,
                    noise=noise))
            elif isinstance(noise, Iterable):
                rdms.append(calc_rdm_movie(
                    dataset[i_dat], method=method,
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
            dat_single = dat.convert_to_dataset(time_descriptor)
            rdms.append(calc_rdm(dat_single, method=method,
                                 descriptor=descriptor, noise=noise,
                                 cv_descriptor=cv_descriptor,
                                 prior_lambda=prior_lambda,
                                 prior_weight=prior_weight))

        rdm = concat(rdms)
        rdm.rdm_descriptors[time_descriptor] = time
    return rdm


def calc_rdm_euclid(dataset, descriptor=None):
    """
    Args:
        dataset (rsatoolbox.data.DatasetBase):
            The dataset the RDM is computed from
        descriptor (String):
            obs_descriptor used to define the rows/columns of the RDM
            defaults to one row/column per row in the dataset
    Returns:
        rsatoolbox.rdm.rdms.RDMs: RDMs object with the one RDM
    """

    measurements, desc, descriptor = _parse_input(dataset, descriptor)
    sum_sq_measurements = np.sum(measurements**2, axis=1, keepdims=True)
    rdm = sum_sq_measurements + sum_sq_measurements.T \
        - 2 * np.dot(measurements, measurements.T)
    rdm = _extract_triu_(rdm) / measurements.shape[1]
    rdm = RDMs(dissimilarities=np.array([rdm]),
               dissimilarity_measure='squared euclidean',
               rdm_descriptors=deepcopy(dataset.descriptors))
    rdm.pattern_descriptors[descriptor] = desc
    return rdm


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
    ma, desc, descriptor = _parse_input(dataset, descriptor)
    ma = ma - ma.mean(axis=1, keepdims=True)
    ma /= np.sqrt(np.einsum('ij,ij->i', ma, ma))[:, None]
    rdm = 1 - np.einsum('ik,jk', ma, ma)
    rdm = RDMs(dissimilarities=np.array([rdm]),
               dissimilarity_measure='correlation',
               rdm_descriptors=deepcopy(dataset.descriptors))
    rdm.pattern_descriptors[descriptor] = desc
    return rdm


def calc_rdm_mahalanobis(dataset, descriptor=None, noise=None):
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

    Returns:
        rsatoolbox.rdm.rdms.RDMs: RDMs object with the one RDM

    """
    if noise is None:
        rdm = calc_rdm_euclid(dataset, descriptor)
    else:
        measurements, desc, descriptor = _parse_input(dataset, descriptor)
        noise = _check_noise(noise, dataset.n_channel)
        kernel = measurements @ noise @ measurements.T
        rdm = np.expand_dims(np.diag(kernel), 0) + np.expand_dims(np.diag(kernel), 1)\
            - 2 * kernel
        rdm = _extract_triu_(rdm) / measurements.shape[1]
        rdm = RDMs(dissimilarities=np.array([rdm]),
                   dissimilarity_measure='squared mahalanobis',
                   rdm_descriptors=deepcopy(dataset.descriptors))
        rdm.pattern_descriptors[descriptor] = desc
        rdm.descriptors['noise'] = noise
    return rdm


def calc_rdm_crossnobis(dataset, descriptor, noise=None,
                        cv_descriptor=None):
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

    Returns:
        rsatoolbox.rdm.rdms.RDMs: RDMs object with the one RDM

    """
    noise = _check_noise(noise, dataset.n_channel)
    if noise is None:
        noise = np.eye(dataset.n_channel)
    if descriptor is None:
        raise ValueError('descriptor must be a string! Crossvalidation' +
                         'requires multiple measurements to be grouped')
    if cv_descriptor is None:
        cv_desc = _gen_default_cv_descriptor(dataset, descriptor)
        dataset.obs_descriptors['cv_desc'] = cv_desc
        cv_descriptor = 'cv_desc'
    dataset.sort_by(descriptor)
    cv_folds = np.unique(np.array(dataset.obs_descriptors[cv_descriptor]))
    rdms = []
    if (noise is None) or (isinstance(noise, np.ndarray) and noise.ndim == 2):
        for i_fold in range(len(cv_folds)):
            fold = cv_folds[i_fold]
            data_test = dataset.subset_obs(cv_descriptor, fold)
            data_train = dataset.subset_obs(cv_descriptor,
                                            np.setdiff1d(cv_folds, fold))
            measurements_train, _, _ = \
                average_dataset_by(data_train, descriptor)
            measurements_test, _, _ = \
                average_dataset_by(data_test, descriptor)
            rdm = _calc_rdm_crossnobis_single(
                measurements_train, measurements_test, noise)
            rdms.append(rdm)
    else:  # a list of noises was provided
        measurements = []
        variances = []
        for i_fold in range(len(cv_folds)):
            data = dataset.subset_obs(cv_descriptor, cv_folds[i_fold])
            measurements.append(average_dataset_by(data, descriptor)[0])
            variances.append(np.linalg.inv(noise[i_fold]))
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
    rdm = RDMs(dissimilarities=np.array([rdm]),
               dissimilarity_measure='crossnobis',
               rdm_descriptors=deepcopy(dataset.descriptors))
    _, desc, _ = average_dataset_by(dataset, descriptor)
    rdm.pattern_descriptors[descriptor] = desc
    rdm.descriptors['noise'] = noise
    rdm.descriptors['cv_descriptor'] = cv_descriptor
    return rdm


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
    measurements, desc, descriptor = _parse_input(dataset, descriptor)
    measurements = (measurements + prior_lambda * prior_weight) \
        / (1 + prior_weight)
    kernel = measurements @ np.log(measurements).T
    rdm = np.expand_dims(np.diag(kernel), 0) + np.expand_dims(np.diag(kernel), 1)\
        - kernel - kernel.T
    rdm = _extract_triu_(rdm) / measurements.shape[1]
    rdm = RDMs(dissimilarities=np.array([rdm]),
               dissimilarity_measure='poisson',
               rdm_descriptors=deepcopy(dataset.descriptors))
    rdm.pattern_descriptors[descriptor] = desc
    return rdm


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
        rdm = np.expand_dims(np.diag(kernel), 0) + np.expand_dims(np.diag(kernel), 1)\
            - kernel - kernel.T
        rdm = _extract_triu_(rdm) / measurements_train.shape[1]
    rdm = RDMs(dissimilarities=np.array([rdm]),
               dissimilarity_measure='poisson_cv',
               rdm_descriptors=deepcopy(dataset.descriptors))
    _, desc, _ = average_dataset_by(dataset, descriptor)
    rdm.pattern_descriptors[descriptor] = desc
    return rdm


def _calc_rdm_crossnobis_single(measurements1, measurements2, noise):
    kernel = measurements1 @ noise @ measurements2.T
    rdm = np.expand_dims(np.diag(kernel), 0) + np.expand_dims(np.diag(kernel), 1)\
        - kernel - kernel.T
    return _extract_triu_(rdm) / measurements1.shape[1]


def _gen_default_cv_descriptor(dataset, descriptor):
    """ generates a default cv_descriptor for crossnobis
    This assumes that the first occurence each descriptor value forms the
    first group, the second occurence forms the second group, etc.
    """
    desc = dataset.obs_descriptors[descriptor]
    values, counts = np.unique(desc, return_counts=True)
    assert np.all(counts == counts[0]), (
        'cv_descriptor generation failed:\n'
        + 'different number of observations per pattern')
    n_repeats = counts[0]
    cv_descriptor = np.zeros_like(desc)
    for i_val in values:
        cv_descriptor[desc == i_val] = np.arange(n_repeats)
    return cv_descriptor


def _calc_pairwise_differences(measurements):
    n, m = measurements.shape
    diff = np.zeros((int(n * (n - 1) / 2), m))
    k = 0
    for i in range(measurements.shape[0]):
        for j in range(i+1, measurements.shape[0]):
            diff[k] = measurements[i] - measurements[j]
            k += 1
    return diff


def _parse_input(dataset, descriptor):
    if descriptor is None:
        measurements = dataset.measurements
        desc = np.arange(measurements.shape[0])
        descriptor = 'pattern'
    else:
        measurements, desc, _ = average_dataset_by(dataset, descriptor)
    return measurements, desc, descriptor


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
    elif isinstance(noise, Iterable):
        for i in range(len(noise)):
            noise[i] = _check_noise(noise[i], n_channel)
    elif isinstance(noise, dict):
        for key in noise.keys():
            noise[key] = _check_noise(noise[key], n_channel)
    else:
        raise ValueError('noise(s) must have shape n_channel x n_channel')
    return noise
