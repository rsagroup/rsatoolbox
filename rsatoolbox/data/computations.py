#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dataset computations
"""

import numpy as np
from rsatoolbox.util.data_utils import get_unique_inverse


def average_dataset(dataset):
    """
    computes the average of a dataset

    Args:
        dataset(rsatoolbox.data.Dataset): the dataset to operate on

    Returns:
        numpy.ndarray: average: average activation vector
    """
    return np.mean(dataset.measurements, axis=0)


def average_dataset_by_slow(dataset, by):
    """
    computes the average of a dataset per value of a descriptor

    Args:
        dataset(rsatoolbox.data.Dataset): the dataset to operate on
        by(String): which obs_descriptor to split by

    Returns:
        numpy.ndarray: average: average activation vector
    """
    datasets = dataset.split_obs(by)
    descriptor = [d.obs_descriptors[by][0] for d in datasets]
    average = [average_dataset(d) for d in datasets]
    n_obs = [d.n_obs for d in datasets]
    return np.array(average), descriptor, n_obs


def average_dataset_by(dataset, by):
    """
    computes the average of a dataset per value of a descriptor

    Args:
        dataset(rsatoolbox.data.Dataset): the dataset to operate on
        by(String): which obs_descriptor to split by

    Returns:
        numpy.ndarray: average: average activation vector
    """
    unique_values, inverse = get_unique_inverse(dataset.obs_descriptors[by])
    average = np.nan * np.empty(
        (len(unique_values), dataset.measurements.shape[1]))
    n_obs = np.nan * np.empty(len(unique_values))
    for i_v, v in enumerate(unique_values):
        measurements = dataset.measurements[inverse == i_v, :]
        average[i_v] = np.mean(measurements, axis=0)
        n_obs[i_v] = measurements.shape[0]
    return average, unique_values, n_obs
