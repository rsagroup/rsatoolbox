#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dataset computations
@author: heiko
"""

import numpy as np


def average_dataset(dataset):
    """
    computes the average of a dataset

    Args:
        dataset(pyrsa.data.Dataset): the dataset to operate on

    Returns:
        numpy.ndarray: average: average activation vector
    """
    return np.mean(dataset.measurements, axis=0)


def average_dataset_by(dataset, by):
    """
    computes the average of a dataset per value of a descriptor

    Args:
        dataset(pyrsa.data.Dataset): the dataset to operate on
        by(String): which obs_descriptor to split by

    Returns:
        numpy.ndarray: average: average activation vector
    """
    datasets = dataset.split_obs(by)
    descriptor = [d.obs_descriptors[by][0] for d in datasets]
    average = [average_dataset(d) for d in datasets]
    n_obs = [d.n_obs for d in datasets]
    return np.array(average), descriptor, n_obs
