#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Definition of RSA Disk Dataset class
"""

from __future__ import annotations
from typing import List, Optional
from warnings import warn
from copy import deepcopy
import numpy as np
from h5py import File, Group, Empty
from rsatoolbox.data.base import DatasetBase


class DatasetDisk(DatasetBase):

    """
    DatasetDisk is a class of Dataset that is stored in a hdf5 file on disk.
    This is meant for very large datasets that cannot be kept in memory effectively.
    This occurs in searchlight analyses or any whole brain fMRI analyses.

    Note that datasets of this size will usually be too large to compute anything, too!
    -> Do subset this kind of dataset before trying to compute RDMs or similar steps.

    See demo_XXX for an example how to use this class.
    """

    def __init__(self, measurements, descriptors=None,
                 obs_descriptors=None, channel_descriptors=None,
                 check_dims=True):
        if measurements.ndim != 2:
            raise AttributeError(
                "measurements must be in dimension n_obs x n_channel")
        self.measurements = measurements
        self.n_obs, self.n_channel = self.measurements.shape
        if check_dims:
            check_descriptor_length_error(obs_descriptors,
                                          "obs_descriptors",
                                          self.n_obs
                                          )
            check_descriptor_length_error(channel_descriptors,
                                          "channel_descriptors",
                                          self.n_channel
                                          )
        self.descriptors = parse_input_descriptor(descriptors)
        self.obs_descriptors = parse_input_descriptor(obs_descriptors)
        self.channel_descriptors = parse_input_descriptor(channel_descriptors)

    def __repr__(self):
        """
        defines string which is printed for the object
        """
        return (f'rsatoolbox.data.{self.__class__.__name__}(\n'
                f'measurements = \n{self.measurements}\n'
                f'descriptors = \n{self.descriptors}\n'
                f'obs_descriptors = \n{self.obs_descriptors}\n'
                f'channel_descriptors = \n{self.channel_descriptors}\n'
                )

    def __str__(self):
        """
        defines the output of print
        """
        string_desc = format_descriptor(self.descriptors)
        string_obs_desc = format_descriptor(self.obs_descriptors)
        string_channel_desc = format_descriptor(self.channel_descriptors)
        if self.measurements.shape[0] > 5:
            measurements = self.measurements[:5, :]
        else:
            measurements = self.measurements
        return (f'rsatoolbox.data.{self.__class__.__name__}\n'
                f'measurements = \n{measurements}\n...\n\n'
                f'descriptors: \n{string_desc}\n\n'
                f'obs_descriptors: \n{string_obs_desc}\n\n'
                f'channel_descriptors: \n{string_channel_desc}\n'
                )

    def __eq__(self, other: Dataset) -> bool:
        """Test for equality
        This magic method gets called when you compare two
        Datasets objects: `ds1 == ds2`.
        True if the objects are of the same type, and
        measurements and descriptors are equal.

        Args:
            other (Dataset): The second Dataset to compare to

        Returns:
            bool: True if the objects' properties are equal
        """
        return all([
            isinstance(other, Dataset),
            np.all(self.measurements == other.measurements),
            self.descriptors == other.descriptors,
            desc_eq(self.obs_descriptors, other.obs_descriptors),
            desc_eq(self.channel_descriptors, other.channel_descriptors),
        ])

    def copy(self) -> Dataset:
        """Return a copy of this object, with all properties
        equal to the original's

        Returns:
            Dataset: Value copy
        """
        return Dataset(
            measurements=self.measurements.copy(),
            descriptors=deepcopy(self.descriptors),
            obs_descriptors=deepcopy(self.obs_descriptors),
            channel_descriptors=deepcopy(self.channel_descriptors)
        )

    def split_obs(self, by):
        """ Returns a list Datasets splited by obs

        Args:
            by(String): the descriptor by which the splitting is made

        Returns:
            list of Datasets, split by the selected obs_descriptor
        """
        unique_values, inverse = get_unique_inverse(self.obs_descriptors[by])
        dataset_list = []
        for i_v, _ in enumerate(unique_values):
            selection = np.where(inverse == i_v)[0]
            measurements = self.measurements[selection, :]
            descriptors = self.descriptors.copy()
            obs_descriptors = subset_descriptor(
                self.obs_descriptors, selection)
            channel_descriptors = self.channel_descriptors
            dataset = Dataset(measurements=measurements,
                              descriptors=descriptors,
                              obs_descriptors=obs_descriptors,
                              channel_descriptors=channel_descriptors,
                              check_dims=False)
            dataset_list.append(dataset)
        return dataset_list

    def split_channel(self, by):
        """ Returns a list Datasets splited by channels

        Args:
            by(String): the descriptor by which the split is done

        Returns:
            list of Datasets,  split by the selected channel_descriptor
        """
        unique_values, inverse = get_unique_inverse(self.channel_descriptors[by])
        dataset_list = []
        for i_v, v in enumerate(unique_values):
            selection = np.where(inverse == i_v)[0]
            measurements = self.measurements[:, selection]
            descriptors = self.descriptors.copy()
            descriptors[by] = v
            obs_descriptors = self.obs_descriptors
            channel_descriptors = subset_descriptor(
                self.channel_descriptors, selection)
            dataset = Dataset(measurements=measurements,
                              descriptors=descriptors,
                              obs_descriptors=obs_descriptors,
                              channel_descriptors=channel_descriptors,
                              check_dims=False)
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
        selection = num_index(self.obs_descriptors[by], value)
        measurements = self.measurements[selection, :]
        descriptors = self.descriptors
        obs_descriptors = subset_descriptor(
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
        selection = num_index(self.channel_descriptors[by], value)
        measurements = self.measurements[:, selection]
        descriptors = self.descriptors
        obs_descriptors = self.obs_descriptors
        channel_descriptors = subset_descriptor(
            self.channel_descriptors, selection)
        dataset = Dataset(measurements=measurements,
                          descriptors=descriptors,
                          obs_descriptors=obs_descriptors,
                          channel_descriptors=channel_descriptors)
        return dataset

    def sort_by(self, by):
        """ sorts the dataset by a given observation descriptor

        Args:
            by(String): the descriptor by which the dataset shall be sorted

        Returns:
            ---

        """
        desc = self.obs_descriptors[by]
        order = np.argsort(desc, kind='stable')
        self.measurements = self.measurements[order]
        self.obs_descriptors = subset_descriptor(self.obs_descriptors, order)

    def get_measurements(self):
        "Getter function for measurements"
        return self.measurements.copy()

    def get_measurements_tensor(self, by):
        """ Returns a tensor version of the measurements array, split by an
        observation descriptor. This procedure will keep the order of
        measurements the same as it is in the dataset.

        Args:
            by(String):
                the descriptor by which the splitting is made

        Returns:
            measurements_tensor (numpy.ndarray):
                n_obs_rest x n_channel x n_obs_by 3d-array, where n_obs_by is
                are the unique values that the obs_descriptor "by" takes, and
                n_obs_rest is the remaining number of observations per unique
                instance of "by"

        """
        assert by in self.obs_descriptors.keys(), \
            "third dimension not in obs_descriptors"
        unique_values = get_unique_unsorted(self.obs_descriptors[by])
        measurements_list = []
        for v in unique_values:
            selection = np.array([desc == v
                                  for desc in self.obs_descriptors[by]])
            measurements_subset = self.measurements[selection, :]
            measurements_list.append(measurements_subset)
        measurements_tensor = np.stack(measurements_list, axis=0)
        measurements_tensor = np.swapaxes(measurements_tensor, 1, 2)
        return measurements_tensor, unique_values

    def odd_even_split(self, obs_desc):
        """
        Perform a simple odd-even split on an rsatoolbox dataset. It will be
        partitioned into n different datasets, where n is the number of
        distinct values on dataset.obs_descriptors[obs_desc].
        The resulting list will be split into odd and even (index) subset.
        The datasets contained in these subsets will then be merged.

        Args:
            obs_desc (str):
                Observation descriptor, basis for partitioning (must contained
                in keys of dataset.obs_descriptors)

        Returns:
            odd_split (Dataset):
                subset of the Dataset with odd list-indices after partitioning
                according to obs_desc
            even_split (Dataset):
                subset of the Dataset with even list-indices after partitioning
                according to obs_desc
        """
        assert obs_desc in self.obs_descriptors.keys(), \
            "obs_desc must be contained in keys of dataset.obs_descriptors"
        ds_part = self.split_obs(obs_desc)
        odd_list = ds_part[0::2]
        even_list = ds_part[1::2]
        odd_split = merge_datasets(odd_list)
        even_split = merge_datasets(even_list)
        return odd_split, even_split

    def nested_odd_even_split(self, l1_obs_desc, l2_obs_desc):
        """
        Nested version of odd_even_split, where dataset is first partitioned
        according to the l1_obs_desc and each partition is again partitioned
        according to the l2_obs_desc (after which the actual oe-split occurs).

        Useful for balancing, especially if the order of your measurements is
        inconsistent, or if the two descriptors are not orthogonalized. It's
        advised to apply .sort_by(l2_obs_desc) to the output of this function.

        Args:
            l1_obs_desc (str):
                Observation descriptor, basis for level 1 partitioning
                (must contained in keys of dataset.obs_descriptors)

        Returns:
            odd_split (Dataset):
                subset of the Dataset with odd list-indices after partitioning
                according to obs_desc
            even_split (Dataset):
                subset of the Dataset with even list-indices after partitioning
                according to obs_desc

        """
        assert l1_obs_desc and l2_obs_desc in self.obs_descriptors.keys(), \
            "observation descriptors must be contained in keys " \
            + "of dataset.obs_descriptors"
        ds_part = self.split_obs(l1_obs_desc)
        odd_list = []
        even_list = []
        for partition in ds_part:
            odd_split, even_split = partition.odd_even_split(l2_obs_desc)
            odd_list.append(odd_split)
            even_list.append(even_split)
        odd_split = merge_datasets(odd_list)
        even_split = merge_datasets(even_list)
        return odd_split, even_split

    @staticmethod
    def from_df(df: DataFrame,
                channels: Optional[List] = None,
                channel_descriptor: Optional[str] = None) -> Dataset:
        """Create a Dataset from a Pandas DataFrame

        Float columns are interpreted as channels, and their names stored as a
        channel descriptor "name".
        Columns of any other datatype will be interpreted as observation
        descriptors, unless they have the same value throughout,
        in which case they will be interpreted as Dataset descriptor.

        Args:
            df (DataFrame): a long-format DataFrame
            channels (list): list of column names to interpret as channels.
                By default all float columns are considered channels.
            channel_descriptor (str): Name of the channel descriptor to create
                on the Dataset which contains the column names.
                Default is "name".

        Returns:
            Dataset: Dataset representing the data from the DataFrame
        """
        if channels is None:
            channels = [c for (c, t) in df.dtypes.items() if 'float' in str(t)]
        if channel_descriptor is None:
            channel_descriptor = 'name'
        descriptors = set(df.columns).difference(channels)
        ds_descriptors, obs_descriptors = dict(), dict()
        for desc in descriptors:
            if df[desc].unique().size == 1:
                ds_descriptors[desc] = df[desc][0]
            else:
                obs_descriptors[desc] = list(df[desc])
        return Dataset(
            measurements=df[channels].values,
            descriptors=ds_descriptors,
            obs_descriptors=obs_descriptors,
            channel_descriptors={channel_descriptor: channels}
        )

    def to_df(self, channel_descriptor: Optional[str] = None) -> DataFrame:
        """returns a Pandas DataFrame representing this Dataset

        Channels, observation descriptors and Dataset descriptors make up the
        columns. Rows represent observations.

        Note that channel descriptors beyond the one used for the column names
        will not be represented.

        Args:
            channel_descriptor: Which channel descriptor to use to
                label the data columns in the Dataframe. Defaults to the
                first channel descriptor.

        Returns:
            DataFrame: A pandas DataFrame representing the Dataset
        """
        desc = channel_descriptor or list(self.channel_descriptors.keys())[0]
        ch_names = self.channel_descriptors[desc]
        df = DataFrame(self.measurements, columns=ch_names)
        all_descriptors = {**self.obs_descriptors, **self.descriptors}
        for dname, dval in all_descriptors.items():
            df[dname] = dval
        return df
