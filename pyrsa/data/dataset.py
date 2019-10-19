#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Definition of RSA Dataset class and subclasses
@author: baihan, jdiedrichsen
"""

import numpy as np
import pyrsa as rsa


class DatasetBase:
    """
    Abstract dataset class.
    Defines members that every class needs to have, but does not
    implement any interesting behavior. Inherit from this class
    to define specific dataset types

        Args:
            measurements (numpy.ndarray): n_obs x n_channel 2d-array,
                                          or n_set x n_obs x n_channel
                                          3d-array
            descriptors (dict):           descriptors with 1 value per
                                          Dataset object
            obs_descriptors (dict):       observation descriptors (all
                                          are array-like with shape =
                                          (n_obs,...))
            channel_descriptors (dict):   channel descriptors (all are
                                          array-like with shape =
                                          (n_channel,...))
        Returns:
            dataset object
    """
    def __init__(self, measurements=None, descriptors=None,
                 obs_descriptors=None, channel_descriptors=None):
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

    def split_obs(self, by):
        """ Returns a list Datasets splited by obs
        Args:
            by(String): the descriptor by which the splitting is made

        Returns:
            list of Datasets, splitted by the selected obs_descriptor
        """
        raise NotImplementedError(
            "split_obs function not implemented in used Dataset class!"
        )

    def split_channel(self, by):
        """ Returns a list Datasets splited by channels
        Args:
            by(String): the descriptor by which the splitting is made

        Returns:
            list of Datasets,  splitted by the selected channel_descriptor
        """
        raise NotImplementedError(
            "split_channel function not implemented in used Dataset class!"
        )

    def subset_obs(self, by, value):
        """ Returns a subsetted Dataset defined by certain obs value
        Args:
            by(String): the descriptor by which the subset selection is made
                        from obs dimension
            value:      the value by which the subset selection is made
                        from obs dimension

        Returns:
            Dataset, with subset defined by the selected obs_descriptor
        """
        raise NotImplementedError(
            "subset_obs function not implemented in used Dataset class!"
        )

    def subset_channel(self, by, value):
        """ Returns a subsetted Dataset defined by certain channel value
        Args:
            by(String): the descriptor by which the subset selection is made
                        from channel dimension
            value:      the value by which the subset selection is made
                        from channel dimension

        Returns:
            Dataset, with subset defined by the selected channel_descriptor
        """
        raise NotImplementedError(
            "subset_channel function not implemented in used Dataset class!"
        )


class Dataset(DatasetBase):
    """
    Dataset class is a standard version of DatasetBase.
    It contains one data set - or multiple data sets with the same structure
    """
    def split_obs(self, by):
        """ Returns a list Datasets splited by obs
        Args:
            by(String): the descriptor by which the splitting is made

        Returns:
            list of Datasets, splitted by the selected obs_descriptor
        """
        unique_values = set(self.obs_descriptors[by])
        dataset_list = []
        for v in unique_values:
            dataset_list.append(
                self.measurements[:, self.obs_descriptors[by] == v, :])
        return dataset_list
        # TODO: for 3d measurements, need implementations.

    def split_channel(self, by):
        """ Returns a list Datasets splited by channels
        Args:
            by(String): the descriptor by which the splitting is made

        Returns:
            list of Datasets,  splitted by the selected channel_descriptor
        """
        unique_values = set(self.channel_descriptors[by])
        dataset_list = []
        for v in unique_values:
            dataset_list.append(
                self.measurements[:, :, self.channel_descriptors[by] == v])
        return dataset_list
        # TODO: for 3d measurements, need implementations.

    def subset_obs(self, by, value):
        """ Returns a subsetted Dataset defined by certain obs value
        Args:
            by(String): the descriptor by which the subset selection
                        is made from obs dimension
            value:      the value by which the subset selection is made
                        from obs dimension

        Returns:
            Dataset, with subset defined by the selected obs_descriptor
        """
        return self.measurements[:, self.obs_descriptors[by] == value, :]
        # TODO: for 3d measurements, need implementations.

    def subset_channel(self, by, value):
        """ Returns a subsetted Dataset defined by certain channel value
        Args:
            by(String): the descriptor by which the subset selection is
                        made from channel dimension
            value:      the value by which the subset selection is made
                        from channel dimension

        Returns:
            Dataset, with subset defined by the selected channel_descriptor
        """
        return self.measurements[:, :, self.channel_descriptors[by] == value]
        # TODO: for 3d measurements, need implementations.
