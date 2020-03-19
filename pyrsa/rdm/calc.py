#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calculation of RDMs from datasets
@author: heiko
"""

import numpy as np
from collections.abc import Iterable
from pyrsa.rdm.rdms import RDMs
from pyrsa.rdm.rdms import concat
from pyrsa.data.dataset import Dataset
from pyrsa.data import average_dataset_by
from pyrsa.util.matrix import pairwise_contrast_sparse


def calc_rdm(dataset, method='euclidean', descriptor=None, noise=None):
    """
    calculates an RDM from an input dataset

    Args:
        dataset (pyrsa.data.dataset.DatasetBase):
            The dataset the RDM is computed from
        method (String):
            a description of the dissimilarity measure (e.g. 'Euclidean')
        descriptor (String):
            obs_descriptor used to define the rows/columns of the RDM
        noise (numpy.ndarray):
            dataset.n_channel x dataset.n_channel
            precision matrix used to calculate the RDM
            used only for Mahalanobis and Crossnobis estimators

    Returns:
        pyrsa.rdm.rdms.RDMs: RDMs object with the one RDM

    """
    if isinstance(dataset, Iterable):
        rdms = []
        for dat in dataset:
            rdms.append(calc_rdm(dat, method=method, descriptor=descriptor,
                                 noise=noise))
        rdm = concat(rdms)
    else:
        if method == 'euclidean':
            rdm = calc_rdm_euclid(dataset, descriptor)
        elif method == 'correlation':
            rdm = calc_rdm_correlation(dataset, descriptor)
        elif method == 'mahalanobis':
            rdm = calc_rdm_mahalanobis(dataset, descriptor, noise)
        elif method == 'crossnobis':
            rdm = calc_rdm_crossnobis(dataset, descriptor, noise)
        else:
            raise(NotImplementedError)
    return rdm


def calc_rdm_euclid(dataset, descriptor=None):
    """
    calculates an RDM from an input dataset using euclidean distance
    If multiple instances of the same condition are found in the dataset
    they are averaged.

    Args:
        dataset (pyrsa.data.DatasetBase):
            The dataset the RDM is computed from
        descriptor (String):
            obs_descriptor used to define the rows/columns of the RDM
            defaults to one row/column per row in the dataset

    Returns:
        pyrsa.rdm.rdms.RDMs: RDMs object with the one RDM

    """
    measurements, desc, descriptor = _parse_input(dataset, descriptor)
    c_matrix = pairwise_contrast_sparse(np.arange(measurements.shape[0]))
    diff = c_matrix @ measurements
    rdm = np.einsum('ij,ij->i', diff, diff) / measurements.shape[1]
    rdm = RDMs(dissimilarities=np.array([rdm]),
               dissimilarity_measure='euclidean',
               descriptors=dataset.descriptors)
    rdm.pattern_descriptors[descriptor] = desc
    return rdm


def calc_rdm_correlation(dataset, descriptor=None):
    """
    calculates an RDM from an input dataset using correlation distance
    If multiple instances of the same condition are found in the dataset
    they are averaged.

    Args:
        dataset (pyrsa.data.DatasetBase):
            The dataset the RDM is computed from
        descriptor (String):
            obs_descriptor used to define the rows/columns of the RDM
            defaults to one row/column per row in the dataset

    Returns:
        pyrsa.rdm.rdms.RDMs: RDMs object with the one RDM

    """
    ma, desc, descriptor = _parse_input(dataset, descriptor)

    ma = ma - ma.mean(axis=1, keepdims=True)
    ma /= np.sqrt(np.einsum('ij,ij->i', ma, ma))[:, None]
    rdm = 1 - np.einsum('ik,jk', ma, ma)
    rdm = RDMs(dissimilarities=np.array([rdm]),
               dissimilarity_measure='correlation',
               descriptors=dataset.descriptors)
    rdm.pattern_descriptors[descriptor] = desc
    return rdm


def calc_rdm_mahalanobis(dataset, descriptor=None, noise=None):
    """
    calculates an RDM from an input dataset using mahalanobis distance
    If multiple instances of the same condition are found in the dataset
    they are averaged.

    Args:
        dataset (pyrsa.data.dataset.DatasetBase):
            The dataset the RDM is computed from
        descriptor (String):
            obs_descriptor used to define the rows/columns of the RDM
            defaults to one row/column per row in the dataset
        noise (numpy.ndarray):
            dataset.n_channel x dataset.n_channel
            precision matrix used to calculate the RDM

    Returns:
        pyrsa.rdm.rdms.RDMs: RDMs object with the one RDM

    """
    measurements, desc, descriptor = _parse_input(dataset, descriptor)
    noise = _check_noise(noise, dataset.n_channel)
    c_matrix = pairwise_contrast_sparse(np.arange(measurements.shape[0]))
    diff = c_matrix @ measurements
    diff2 = (noise @ diff.T).T
    rdm = np.einsum('ij,ij->i', diff, diff2) / measurements.shape[1]
    rdm = RDMs(dissimilarities=np.array([rdm]),
               dissimilarity_measure='Mahalanobis',
               descriptors=dataset.descriptors)
    rdm.pattern_descriptors[descriptor] = desc
    rdm.descriptors['noise'] = noise
    return rdm


def calc_rdm_crossnobis(dataset,
                        descriptor,
                        noise=None,
                        cv_descriptor=None
                        ):
    """
    calculates an RDM from an input dataset using Cross-nobis distance
    This performs leave one out crossvalidation over the cv_descriptor

    Args:
        dataset (pyrsa.data.dataset.DatasetBase):
            The dataset the RDM is computed from
        descriptor (String):
            obs_descriptor used to define the rows/columns of the RDM
            defaults to one row/column per row in the dataset
        noise (numpy.ndarray):
            dataset.n_channel x dataset.n_channel
            precision matrix used to calculate the RDM
        cv_descriptor (String):
            obs_descriptor which determines the cross-validation folds

    Returns:
        pyrsa.rdm.rdms.RDMs: RDMs object with the one RDM

    """
    noise = _check_noise(noise, dataset.n_channel)
    if descriptor is None:
        raise ValueError('descriptor must be a string! Crossvalidation' +
                         'requires multiple measurements to be grouped')
    cv_folds = np.unique(np.array(dataset.obs_descriptors[cv_descriptor]))
    weights = []
    rdms = []
    for i_fold in cv_folds:
        data_train = dataset.subset_obs(cv_descriptor, i_fold)
        data_test = dataset.subset_obs(cv_descriptor,
                                       np.setdiff1d(cv_folds, i_fold))
        measurements_train, desc = average_dataset_by(data_train, descriptor)
        measurements_test, desc = average_dataset_by(data_test, descriptor)
        rdm = _calc_rdm_crossnobis_single(measurements_train,
                                          measurements_test,
                                          noise)
        rdms.append(rdm)
        weights.append(data_test.n_obs)
    rdms = np.array(rdms)
    weights = np.array(weights)
    rdm = np.einsum('ij,i->j', rdms, weights) / np.sum(weights)
    rdm = RDMs(dissimilarities=np.array([rdm]),
               dissimilarity_measure='crossnobis',
               descriptors=dataset.descriptors)
    if descriptor is None:
        rdm.pattern_descriptors['pattern'] = np.arange(rdm.n_cond)
    else:
        rdm.pattern_descriptors[descriptor] = desc
    rdm.descriptors['noise'] = noise
    rdm.descriptors['cv_descriptor'] = cv_descriptor
    return rdm


def _calc_rdm_crossnobis_single(measurements1, measurements2, noise):
    c_matrix = pairwise_contrast_sparse(np.arange(measurements1.shape[0]))
    diff_1 = c_matrix @ measurements1
    diff_2 = c_matrix @ measurements2
    diff_2 = noise @ diff_2.transpose()
    rdm = np.einsum('kj,jk->k', diff_1, diff_2) / measurements1.shape[1]
    return rdm


def _parse_input(dataset, descriptor):
    if descriptor is None:
        measurements = dataset.measurements
        desc = np.arange(measurements.shape[0])
        descriptor = 'pattern'
    else:
        measurements, desc = average_dataset_by(dataset, descriptor)
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
        noise = np.eye(n_channel)
    elif isinstance(noise, np.ndarray):
        assert noise.ndim == 2
        assert np.all(noise.shape == (n_channel, n_channel))
    else:
        raise ValueError('noise must have shape n_channel x n_channel')
    return noise
