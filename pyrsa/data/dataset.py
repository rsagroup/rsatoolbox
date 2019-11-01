#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Definition of RSA Dataset class and subclasses
@author: baihan, jdiedrichsen
"""

import numpy as np
import pyrsa as rsa
from pyrsa.util.data_utils import check_descriptors_dimension
from pyrsa.util.data_utils import extract_dict
from pyrsa.util.data_utils import get_unique_unsorted


class DatasetBase:
    """
    Abstract dataset class.
    Defines members that every class needs to have, but does not
    implement any interesting behavior. Inherit from this class
    to define specific dataset types

        Args:
            measurements (numpy.ndarray): n_obs x n_channel 2d-array,
            descriptors (dict):           descriptors (metadata)
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
        if measurements is not None:
            if measurements.ndim != 2:
                raise AttributeError(
                    "measurements must be in dimension n_obs x n_channel")
            else:
                self.measurements = measurements
                self.n_obs, self.n_channel = self.measurements.shape
        check_descriptors_dimension(obs_descriptors, "obs_descriptors",
                                    n.n_obs)
        check_descriptors_dimension(channel_descriptors, "channel_descriptors",
                                    n.n_channel)
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
            "split_obs function not implemented in used Dataset class!")

    def split_channel(self, by):
        """ Returns a list Datasets splited by channels
        Args:
            by(String): the descriptor by which the splitting is made

        Returns:
            list of Datasets,  splitted by the selected channel_descriptor
        """
        raise NotImplementedError(
            "split_channel function not implemented in used Dataset class!")

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
            "subset_obs function not implemented in used Dataset class!")

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
            "subset_channel function not implemented in used Dataset class!")


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
        unique_values = get_unique_unsorted(self.obs_descriptors[by])
        dataset_list = []
        for v in unique_values:
            selection = (self.obs_descriptors[by] == v)
            measurements = self.measurements[selection, :]
            descriptors = self.descriptors
            obs_descriptors = extract_dict(
                self.obs_descriptors, selection)
            channel_descriptors = self.channel_descriptors
            dataset = Dataset(measurements=measurements,
                              descriptors=descriptors,
                              obs_descriptors=obs_descriptors,
                              channel_descriptors=channel_descriptors)
            dataset_list.append(dataset)
        return dataset_list

    def split_channel(self, by):
        """ Returns a list Datasets splited by channels
        Args:
            by(String): the descriptor by which the splitting is made

        Returns:
            list of Datasets,  splitted by the selected channel_descriptor
        """
        unique_values = get_unique_unsorted(self.channel_descriptors[by])
        dataset_list = []
        for v in unique_values:
            selection = (self.channel_descriptors[by] == v)
            measurements = self.measurements[:, selection]
            descriptors = self.descriptors
            obs_descriptors = self.obs_descriptors
            channel_descriptors = extract_dict(
                self.channel_descriptors, selection)
            dataset = Dataset(measurements=measurements,
                              descriptors=descriptors,
                              obs_descriptors=obs_descriptors,
                              channel_descriptors=channel_descriptors)
            dataset_list.append(dataset)
        return dataset_list

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
        selection = (self.obs_descriptors[by] == value)
        measurements = self.measurements[selection, :]
        descriptors = self.descriptors
        obs_descriptors = extract_dict(
            self.obs_descriptors, selection)
        channel_descriptors = self.channel_descriptors
        dataset = Dataset(measurements=measurements,
                          descriptors=descriptors,
                          obs_descriptors=obs_descriptors,
                          channel_descriptors=channel_descriptors)
        return dataset

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
        selection = (self.channel_descriptors[by] == value)
        measurements = self.measurements[:, selection]
        descriptors = self.descriptors
        obs_descriptors = self.obs_descriptors
        channel_descriptors = extract_dict(
            self.channel_descriptors, selection)
        dataset = Dataset(measurements=measurements,
                          descriptors=descriptors,
                          obs_descriptors=obs_descriptors,
                          channel_descriptors=channel_descriptors)
        return dataset
