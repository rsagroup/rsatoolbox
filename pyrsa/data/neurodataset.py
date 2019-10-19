#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Definition of RSA Dataset classes for neuroimaging or neural data
@author: baihan
"""
import pyrsa as rsa
from pyrsa.data.dataset import DatasetBase
import numpy as np


def preprocess_fmri(rawdata=None):
    raise NotImplementedError(
        "preprocess_fmri function not implemented!")


class DatasetFmri(DatasetBase):
    """
    DatasetFmri class is a variant of Dataset that takes in raw
    fMRI data and process it into standard Dataset format.
    """
    def __init__(self, rawdata=None, preprocess=preprocess_fmri):

        measurements, descriptors, obs_descriptors, channel_descriptors \
            = preprocess(rawdata)

        if (measurements.ndim == 2):
            self.measurements = measurements
            self.n_set = 1
            self.n_obs, self.n_channel = self.measurements.shape
        elif (measurements.ndim == 3):
            self.measurements = measurements
            self.n_set, self.n_obs, self.n_channel = self.measurements.shape
        self.descriptors = descriptors
        self.obs_descriptors = obs_descriptors
        self.channel_descriptors = channel_descriptors
