#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calculation of RDMs from datasets
@author: heiko
"""

import numpy as np
from pyrsa.rdm.rdms import RDMs
from pyrsa.data.dataset import Dataset
from pyrsa.data import average_dataset_by
from pyrsa.util import contrast_matrix


def calc_rdm(dataset, method='euclidean', descriptor=None, noise=None):
    """
    calculates an RDM from an input dataset

        Args:
            dataset (pyrsa.data.DatasetBase):
                The dataset the RDM is computed from
            method (String):
                a description of the dissimilarity measure (e.g. 'Euclidean')
            descriptor (String):
                obs_descriptor used to define the rows/columns of the RDM
            noise (numpy.ndarray):
                precision matrix used to calculate the RDM
                used only for Mahalanobis and Crossnobis estimators
        Returns:
            RDMs object with the one RDM
    """
    if method == 'mahalanobis':
        rdm = calc_rdm_mahalanobis(dataset, descriptor, noise)
    elif method == 'euclidean':
        rdm = calc_rdm_euclid(dataset)
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
            RDMs object with the one RDM
    """
    if descriptor is None:
        measurements = dataset.measurements
    else:
        measurements,desc = average_dataset_by(dataset, descriptor)
    c_matrix = contrast_matrix(measurements.shape[0])
    diff = np.matmul(c_matrix, measurements)
    rdm = np.einsum('ij,ij->i', diff, diff) / measurements.shape[1]
    rdm = RDMs(dissimilarities = np.array([rdm]), dissimilarity_measure='euclidean',
                 descriptors=dataset.descriptors)
    if descriptor is None:
        rdm.pattern_descriptors['pattern'] = list(np.arange(diff.shape[0]))
    else:
        rdm.pattern_descriptors[descriptor] = desc
    return rdm


def calc_rdm_mahalanobis(dataset, descriptor=None, noise=None):
    """
    calculates an RDM from an input dataset using mahalanobis distance
    If multiple instances of the same condition are found in the dataset
    they are averaged.

        Args:
            dataset (pyrsa.data.DatasetBase):
                The dataset the RDM is computed from
            descriptor (String):
                obs_descriptor used to define the rows/columns of the RDM
                defaults to one row/column per row in the dataset
            noise (numpy.ndarray):
                precision matrix used to calculate the RDM
        Returns:
            RDMs object with the one RDM
    """
    if descriptor is None:
        measurements = dataset.measurements
    else:
        measurements,desc = average_dataset_by(dataset,descriptor)
    if noise is None:
        noise = np.eye(measurements.shape[-1])
    c_matrix = contrast_matrix(measurements.shape[0])
    diff = np.matmul(c_matrix, measurements)
    diff2 = np.matmul(noise,diff.T).T
    rdm = np.einsum('ij,ij->i', diff, diff2) / measurements.shape[1]
    rdm = RDMs(dissimilarities=np.array([rdm]), dissimilarity_measure='Mahalanobis',
                 descriptors=dataset.descriptors)
    if descriptor is None:
        rdm.pattern_descriptors['pattern'] = list(np.arange(diff.shape[0]))
    else:
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
            dataset (pyrsa.data.DatasetBase):
                The dataset the RDM is computed from
            descriptor (String):
                obs_descriptor used to define the rows/columns of the RDM
                defaults to one row/column per row in the dataset
            noise (numpy.ndarray):
                precision matrix used to calculate the RDM
            cv_descriptor (String):
                obs_descriptor which determines the cross-validation folds
                
        Returns:
            RDMs object with the one RDM
    """
    if noise is None:
        noise = np.eye(dataset.n_channel)
    if descriptor is None:
        raise ValueError('descriptor must be a string! Crossvalidation' +
                         'requires multiple measurements to be grouped')    
    cv_folds = np.unique(np.array(dataset.obs_descriptors[cv_descriptor]))
    weights = []
    rdms = []
    for i_fold in cv_folds:
        data_train = dataset.subset_obs(cv_descriptor, i_fold)
        data_test = dataset.subset_obs(cv_descriptor, np.setdiff1d(cv_folds, i_fold))
        measurements_train, desc = average_dataset_by(data_train, descriptor)
        measurements_test, desc = average_dataset_by(data_test, descriptor)
        rdm = calc_rdm_crossnobis_single(measurements_train,
                                         measurements_test,
                                         noise)
        rdms.append(rdm)
        weights.append(data_test.n_obs)
    rdms = np.array(rdms)
    weights = np.array(weights)
    rdm = np.einsum('ij,i->j',rdms,weights)/np.sum(weights)
    rdm = RDMs(dissimilarities = np.array([rdm]), dissimilarity_measure = 'crossnobis',
                 descriptors = dataset.descriptors)
    if descriptor is None:
        rdm.pattern_descriptors['pattern'] = list(np.arange(rdm.n_cond))
    else:
        rdm.pattern_descriptors[descriptor] = desc
    rdm.descriptors['noise'] = noise
    rdm.descriptors['cv_descriptor'] = cv_descriptor
    return rdm


def calc_rdm_crossnobis_single(measurements1,measurements2,noise):
    C = contrast_matrix(measurements1.shape[0])
    diff_1 = np.matmul(C,measurements1)
    diff_2 = np.matmul(C,measurements2)
    diff_2 = np.matmul(noise,diff_2.transpose())
    rdm = np.einsum('kj,jk->k',diff_1,diff_2)/measurements1.shape[1]
    return rdm
