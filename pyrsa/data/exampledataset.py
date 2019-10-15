#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Definition of RSA Dataset classes for arbitrary user-defined data
@author: baihan
"""
import pyrsa as rsa
from pyrsa.data.dataset import DatasetBase
import numpy as np

def preprocess_example(rawdata=None):
    raise NotImplementedError(
            "preprocess_fmri function not implemented!")

class DatasetExample(DatasetBase): 
    """
    DatasetExample class is a variant of Dataset that takes in any arbitrary user defined data and process it into standard Dataset format.
    It also enables the user to specifies arbitrary subset or split methods.
    """
    def __init__(self, rawdata=None,preprocess=preprocess_example):
        measurements,descriptors,obs_descriptors,channel_descriptors=preprocess(rawdata)

        if (measurements.ndim==2):
            self.measurements = measurements
            self.n_set = 1 
            self.n_obs,self.n_channel = self.measurements.shape
        elif (measurements.ndim==3):
            self.measurements = measurements
            self.n_set,self.n_obs,self.n_channel = self.measurements.shape
        self.descriptors = descriptors 
        self.obs_descriptors = obs_descriptors 
        self.channel_descriptors = channel_descriptors 

    def split_obs(self, by):
        """ Returns a list Datasets splited by obs
        Args:
            by(String): the descriptor by which the splitting is made

        Returns:
            list of Datasets, splitted by the selected obs_descriptor
        """
        # TODO
        

    def split_channel(self, by):
        """ Returns a list Datasets splited by channels
        Args:
            by(String): the descriptor by which the splitting is made

        Returns:
            list of Datasets,  splitted by the selected channel_descriptor
        """
        # TODO

    def subset_obs(self, by, value):
        """ Returns a subsetted Dataset defined by certain obs value
        Args:
            by(String): the descriptor by which the subset selection is made from obs dimension
            value: the value by which the subset selection is made from obs dimension

        Returns:
            Dataset, with subset defined by the selected obs_descriptor
        """
        # TODO

    def subset_channel(self, by, value):
        """ Returns a subsetted Dataset defined by certain channel value
        Args:
            by(String): the descriptor by which the subset selection is made from channel dimension
            value: the value by which the subset selection is made from channel dimension

        Returns:
            Dataset, with subset defined by the selected channel_descriptor
        """
        # TODO