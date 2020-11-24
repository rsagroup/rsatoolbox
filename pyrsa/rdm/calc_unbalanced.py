#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calculation of RDMs from datasets
@author: heiko, benjamin
"""

from collections.abc import Iterable
import numpy as np
from copy import deepcopy
from pyrsa.rdm.rdms import RDMs
from pyrsa.rdm.rdms import concat
from pyrsa.data import average_dataset_by
from pyrsa.util.matrix import pairwise_contrast_sparse


def calc_rdm_unbalanced(dataset, method='euclidean', descriptor=None,
                        noise=None, cv_descriptor=None,
                        prior_lambda=1, prior_weight=0.1,
                        weighting='number', enforce_same=False):
    """
    calculates a RDM from an input dataset for unbalanced datasets

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
            defaults to an identity matrix, i.e. euclidean distance

    Returns:
        pyrsa.rdm.rdms.RDMs: RDMs object with the one RDM

    """
    if descriptor is None:
        dataset = deepcopy(dataset)
        dataset.obs_descriptors['index'] = np.arange(dataset.n_obs)
        descriptor = 'index'
    if isinstance(dataset, Iterable):
        rdms = []
        for i_dat in range(len(dataset)):
            if noise is None:
                rdms.append(calc_rdm_unbalanced(
                    dataset[i_dat], method=method, descriptor=descriptor,
                    cv_descriptor=cv_descriptor,
                    prior_lambda=prior_lambda, prior_weight=prior_weight,
                    weighting=weighting, enforce_same=enforce_same))
            elif isinstance(noise, np.ndarray) and noise.ndim == 2:
                rdms.append(calc_rdm_unbalanced(
                    dataset[i_dat], method=method,
                    descriptor=descriptor,
                    noise=noise,
                    cv_descriptor=cv_descriptor,
                    prior_lambda=prior_lambda, prior_weight=prior_weight,
                    weighting=weighting, enforce_same=enforce_same))
            elif isinstance(noise, Iterable):
                rdms.append(calc_rdm_unbalanced(
                    dataset[i_dat], method=method,
                    descriptor=descriptor,
                    noise=noise[i_dat],
                    cv_descriptor=cv_descriptor,
                    prior_lambda=prior_lambda, prior_weight=prior_weight,
                    weighting=weighting, enforce_same=enforce_same))
        rdm = concat(rdms)
    else:
        if method == 'crossnobis' or method == 'poisson_cv':
            rdm = []
            weights = []
            unique_cond = set(dataset.obs_descriptors[descriptor])
            for i, i_des in enumerate(unique_cond):
                for j, j_des in enumerate(unique_cond):
                    if j > i:
                        v, w = calc_one_distance_cv(
                            dataset, descriptor, i_des, j_des,
                            method=method,
                            noise=noise, weighting=weighting,
                            prior_lambda=prior_lambda, 
                            prior_weight=prior_weight,
                            cv_descriptor=cv_descriptor,
                            enforce_same=enforce_same)
                        rdm.append(v)
                        weights.append(w)
        else:
            rdm = []
            weights = []
            unique_cond = set(dataset.obs_descriptors[descriptor])
            for i, i_des in enumerate(unique_cond):
                for j, j_des in enumerate(unique_cond):
                    if j > i:
                        v, w = calc_one_distance(
                            dataset, descriptor, i_des, j_des, method=method,
                            noise=noise, weighting='number')
                        rdm.append(v)
                        weights.append(w)
        rdm = RDMs(
            dissimilarities=np.array([rdm]),
            dissimilarity_measure=method,
            rdm_descriptors=deepcopy(dataset.descriptors))
        rdm.pattern_descriptors[descriptor] = unique_cond
        rdm.rdm_descriptors['weights'] = [weights]
    return rdm



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


def calc_one_distance(dataset, descriptor, i_des, j_des, method='euclidean',
                      noise=None, weighting='number',
                      prior_lambda=1, prior_weight=0.1):
    """
    finds all pairs of vectors to be compared and calculates one distance

    Args:
        dataset (pyrsa.data.DatasetBase):
            dataset to extract from
        descriptor (String):
            key for the descriptor defining the conditions
        i_des : descriptor value
            the value of the first condition
        j_des : descriptor value
            the value of the second condition
        noise : numpy.ndarray (n_channels x n_channels), optional
            the covariance or precision matrix over channels 
            necessary for calculation of mahalanobis distances

    Returns:
        (np.ndarray, np.ndarray) : (value, weight)
        value are the dissimilarities 

    """
    data_i = dataset.subset_obs(descriptor, i_des)
    data_j = dataset.subset_obs(descriptor, j_des)
    values = []
    weights = []
    for vec_i in data_i.measurements:
        for vec_j in data_j.measurements:
            finite = np.isfinite(vec_i) & np.isfinite(vec_j)
            if np.any(finite):
                if weighting == 'number':
                    weight = np.sum(finite)
                elif weighting == 'equal':
                    weight = 1
                dissim = dissimilarity(vec_i[finite], vec_j[finite], method) \
                    / np.sum(finite)
                values.append(dissim)
                weights.append(weight)
    weights = np.array(weights)
    values = np.array(values)
    if np.sum(weights) > 0:
        weight = np.sum(weights)
        value = np.sum(weights * values) / weight
    else:
        value = np.nan
        weight = 0
    return value, weight


def calc_one_distance_cv(dataset, descriptor, i_des, j_des, method='euclidean',
                         noise=None, weighting='number',
                         prior_lambda=1, prior_weight=0.1,
                         cv_descriptor=None, enforce_same=False):
    """
    finds all pairs of vectors to be compared and calculates one distance

    Args:
        dataset (pyrsa.data.DatasetBase):
            dataset to extract from
        descriptor (String):
            key for the descriptor defining the conditions
        i_des : descriptor value
            the value of the first condition
        j_des : descriptor value
            the value of the second condition
        noise : numpy.ndarray (n_channels x n_channels), optional
            the covariance or precision matrix over channels 
            necessary for calculation of mahalanobis distances

    Returns:
        (np.ndarray, np.ndarray) : (value, weight)
        value are the dissimilarities 

    """
    data_i = dataset.subset_obs(descriptor, i_des)
    data_j = dataset.subset_obs(descriptor, j_des)
    values = []
    weights = []
    for i in range(data_i.n_obs):
        for j in range(data_j.n_obs):
            for k in range(i + 1, data_i.n_obs):
                for l in range(j + 1, data_j.n_obs):
                    if cv_descriptor is None:
                        accepted = True
                    else:
                        if (data_i.obs_descriptors[cv_descriptor][i] \
                            == data_i.obs_descriptors[cv_descriptor][k]):
                            accepted = False
                        elif (data_j.obs_descriptors[cv_descriptor][j] \
                            == data_j.obs_descriptors[cv_descriptor][l]):
                            accepted = False
                        elif (data_i.obs_descriptors[cv_descriptor][i] \
                            == data_j.obs_descriptors[cv_descriptor][l]):
                            accepted = False
                        elif (data_j.obs_descriptors[cv_descriptor][j] \
                            == data_i.obs_descriptors[cv_descriptor][k]):
                            accepted = False
                        else:
                            accepted = True
                        if enforce_same:
                            if (data_i.obs_descriptors[cv_descriptor][i] \
                                != data_j.obs_descriptors[cv_descriptor][j]):
                                accepted=False
                            if (data_i.obs_descriptors[cv_descriptor][k] \
                                != data_j.obs_descriptors[cv_descriptor][l]):
                                accepted=False
                    if accepted:
                        vec_i = data_i.measurements[i]
                        vec_j = data_j.measurements[j]
                        vec_k = data_i.measurements[k]
                        vec_l = data_j.measurements[l]
                        finite = np.isfinite(vec_i) & np.isfinite(vec_j) \
                            & np.isfinite(vec_k) & np.isfinite(vec_l)
                        if np.any(finite):
                            if weighting == 'number':
                                weight = np.sum(finite)
                            elif weighting == 'equal':
                                weight = 1
                            dissim = dissimilarity_cv(
                                vec_i[finite], vec_j[finite],
                                vec_k[finite], vec_l[finite],
                                method,
                                noise=noise,
                                prior_lambda=prior_lambda,
                                prior_weight=prior_weight) \
                                / np.sum(finite)
                            values.append(dissim)
                            weights.append(weight)
    weights = np.array(weights)
    values = np.array(values)
    if np.sum(weights) > 0:
        weight = np.sum(weights)
        value = np.sum(weights * values) / weight
    else:
        value = np.nan
        weight = 0
    return value, weight


def dissimilarity(vec_i, vec_j, method, noise=None,
                  prior_lambda=1, prior_weight=0.1):
    if method == 'euclidean':
        dissim = np.sum((vec_i - vec_j) ** 2)
    elif method == 'correlation':
        vec_i = vec_i - np.mean(vec_i)
        vec_j = vec_j - np.mean(vec_j)
        norm_i = np.sum(vec_i ** 2)
        norm_j = np.sum(vec_j ** 2)
        if (norm_i) > 0 and (norm_j > 0):
            dissim = 1 - (np.sum(vec_i * vec_j) 
                          / np.sqrt(norm_i) / np.sqrt(norm_j))
        else:
            dissim = 1
        dissim = dissim * len(vec_i)
    elif method == 'mahalanobis':
        if noise is None:
            dissim = dissimilarity(vec_i, vec_j, 'euclidean')
        else:
            diff = vec_i - vec_j
            diff2 = (noise @ diff.T).T
            dissim = np.sum(diff * diff2)
    elif method == 'poisson':
        vec_i = (vec_i + prior_lambda * prior_weight) \
            / (1 + prior_weight)
        vec_j = (vec_j + prior_lambda * prior_weight) \
            / (1 + prior_weight)
        diff = vec_i - vec_j
        diff_log = np.log(vec_i) - np.log(vec_j)
        dissim = np.sum(diff * diff_log)
    else:
        raise ValueError('dissimilarity method not recognized!')
    return dissim


def dissimilarity_cv(vec_i, vec_j, vec_k, vec_l, method, noise=None,
                     prior_lambda=1, prior_weight=0.1):
    """ helper function for crossvalidated distances 
    """
    if method == 'crossnobis':
        if noise is None:
            diff = vec_i - vec_j
            diff2 = vec_k - vec_l
            dissim = np.sum(diff * diff2)
        else:
            diff = vec_i - vec_j
            diff2 = (noise @ (vec_k - vec_l).T).T
            dissim = np.sum(diff * diff2)
    elif method == 'poisson_cv':
        vec_i = (vec_i + prior_lambda * prior_weight) \
            / (1 + prior_weight)
        vec_j = (vec_j + prior_lambda * prior_weight) \
            / (1 + prior_weight)
        vec_k = (vec_k + prior_lambda * prior_weight) \
            / (1 + prior_weight)
        vec_l = (vec_l + prior_lambda * prior_weight) \
            / (1 + prior_weight)
        diff = vec_i - vec_j
        diff2 = vec_k - vec_l
        diff_log = np.log(vec_i) - np.log(vec_j)
        diff_log2 = np.log(vec_k) - np.log(vec_l)
        dissim = np.sum(diff * diff_log2) + np.sum(diff * diff_log2)
    else:
        raise ValueError('dissimilarity method not recognized!')
    return dissim
