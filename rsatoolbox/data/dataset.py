#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Definition of RSA Dataset class and subclasses

@author: baihan, jdiedrichsen, bpeters, adkipnis
"""

from __future__ import annotations
from typing import List, Optional
import numpy as np
from pandas import DataFrame
from rsatoolbox.util.data_utils import get_unique_unsorted
from rsatoolbox.util.data_utils import get_unique_inverse
from rsatoolbox.util.descriptor_utils import check_descriptor_length_error
from rsatoolbox.util.descriptor_utils import subset_descriptor
from rsatoolbox.util.descriptor_utils import num_index
from rsatoolbox.util.descriptor_utils import format_descriptor
from rsatoolbox.util.descriptor_utils import parse_input_descriptor
from rsatoolbox.util.descriptor_utils import append_obs_descriptors
from rsatoolbox.util.file_io import write_dict_hdf5
from rsatoolbox.util.file_io import write_dict_pkl
from rsatoolbox.util.file_io import read_dict_hdf5
from rsatoolbox.util.file_io import read_dict_pkl
from rsatoolbox.util.file_io import remove_file


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
            are array-like with shape = (n_obs,...))
        channel_descriptors (dict):   channel descriptors (all are
            array-like with shape = (n_channel,...))

    Returns:
        dataset object
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

    def split_obs(self, by):
        """ Returns a list Datasets split by obs

        Args:
            by(String): the descriptor by which the splitting is made

        Returns:
            list of Datasets, splitted by the selected obs_descriptor

        """
        raise NotImplementedError(
            "split_obs function not implemented in used Dataset class!")

    def split_channel(self, by):
        """ Returns a list Datasets split by channels

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

    def save(self, filename, file_type='hdf5', overwrite=False):
        """ Saves the dataset object to a file

        Args:
            filename(String): path to the file
                [or opened file]
            file_type(String): Type of file to create:
                hdf5: hdf5 file
                pkl: pickle file
            overwrite(Boolean): overwrites file if it already exists

        """
        data_dict = self.to_dict()
        if overwrite:
            remove_file(filename)
        if file_type == 'hdf5':
            write_dict_hdf5(filename, data_dict)
        elif file_type == 'pkl':
            write_dict_pkl(filename, data_dict)

    def to_dict(self):
        """ Generates a dictionary which contains the information to
        recreate the dataset object. Used for saving to disc

        Returns:
            data_dict(dict): dictionary with dataset information

        """
        data_dict = {}
        data_dict['measurements'] = self.measurements
        data_dict['descriptors'] = self.descriptors
        data_dict['obs_descriptors'] = self.obs_descriptors
        data_dict['channel_descriptors'] = self.channel_descriptors
        data_dict['type'] = type(self).__name__
        return data_dict

    @staticmethod
    def from_df(
        df: DataFrame,
        channels: Optional[List]=None,
        channel_descriptor: Optional[str]=None) -> Dataset:
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
            Dataset: RSAtoolbox Dataset representing the data from the DataFrame
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

    def to_df(self, channel_descriptor: Optional[str]=None) -> DataFrame:
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
        for dname, dval in {**self.obs_descriptors, **self.descriptors}.items():
            df[dname] = dval
        return df


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
            by(String): the descriptor by which the splitting is made

        Returns:
            list of Datasets,  splitted by the selected channel_descriptor
        """
        unique_values, inverse = get_unique_inverse(self.channel_descriptors[by])
        dataset_list = []
        for i_v, _ in enumerate(unique_values):
            selection = np.where(inverse == i_v)[0]
            measurements = self.measurements[:, selection]
            descriptors = self.descriptors.copy()
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
        order = np.argsort(desc)
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
        odd_split = merge_subsets(odd_list)
        even_split = merge_subsets(even_list)
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
        odd_split = merge_subsets(odd_list)
        even_split = merge_subsets(even_list)
        return odd_split, even_split


class TemporalDataset(Dataset):
    """
    TemporalDataset for spatio-temporal datasets

    Args:
        measurements (numpy.ndarray): n_obs x n_channel x time 3d-array,
        descriptors (dict):           descriptors (metadata)
        obs_descriptors (dict):       observation descriptors (all
            are array-like with shape = (n_obs,...))
        channel_descriptors (dict):   channel descriptors (all are
            array-like with shape = (n_channel,...))
        time_descriptors (dict):      time descriptors (alls are
            array-like with shape= (n_time,...))

            time_descriptors needs to contain one key 'time' that
            specifies the time-coordinate. if None is provided, 'time' is
            set as (0, 1, ..., n_time-1)

    Returns:
        dataset object
    """

    def __init__(self, measurements, descriptors=None,
                 obs_descriptors=None, channel_descriptors=None,
                 time_descriptors=None, check_dims=True):

        if measurements.ndim != 3:
            raise AttributeError(
                "measurements must be in dimension n_obs x n_channel x time")

        self.measurements = measurements
        self.n_obs, self.n_channel, self.n_time = self.measurements.shape

        if time_descriptors is None:
            time_descriptors = {'time': np.arange(self.n_time)}
        elif 'time' not in time_descriptors:
            time_descriptors['time'] = np.arange(self.n_time)
            raise Warning(
                "there was no 'time' provided in dictionary time_descriptors"
                "\n'time' will be set to (0, 1, ..., n_time-1)")

        if check_dims:
            check_descriptor_length_error(obs_descriptors,
                                          "obs_descriptors",
                                          self.n_obs
                                          )
            check_descriptor_length_error(channel_descriptors,
                                          "channel_descriptors",
                                          self.n_channel
                                          )
            check_descriptor_length_error(time_descriptors,
                                          "time_descriptors",
                                          self.n_time
                                          )
        self.descriptors = parse_input_descriptor(descriptors)
        self.obs_descriptors = parse_input_descriptor(obs_descriptors)
        self.channel_descriptors = parse_input_descriptor(channel_descriptors)
        self.time_descriptors = parse_input_descriptor(time_descriptors)

    def __str__(self):
        """
        defines the output of print
        """
        string_desc = format_descriptor(self.descriptors)
        string_obs_desc = format_descriptor(self.obs_descriptors)
        string_channel_desc = format_descriptor(self.channel_descriptors)
        string_time_desc = format_descriptor(self.time_descriptors)
        if self.measurements.shape[0] > 5:
            measurements = self.measurements[:5, :, :]
        else:
            measurements = self.measurements
        return (f'rsatoolbox.data.{self.__class__.__name__}\n'
                f'measurements = \n{measurements}\n...\n\n'
                f'descriptors: \n{string_desc}\n\n'
                f'obs_descriptors: \n{string_obs_desc}\n\n'
                f'channel_descriptors: \n{string_channel_desc}\n'
                f'time_descriptors: \n{string_time_desc}\n'
                )

    def split_obs(self, by):
        """ Returns a list TemporalDataset splited by obs

        Args:
            by(String): the descriptor by which the splitting is made

        Returns:
            list of TemporalDataset, splitted by the selected obs_descriptor
        """
        unique_values, inverse = get_unique_inverse(self.obs_descriptors[by])
        dataset_list = []
        for i_v, _ in enumerate(unique_values):
            selection = np.where(inverse == i_v)[0]
            measurements = self.measurements[selection, :, :]
            descriptors = self.descriptors
            obs_descriptors = subset_descriptor(
                self.obs_descriptors, selection)
            channel_descriptors = self.channel_descriptors
            time_descriptors = self.time_descriptors
            dataset = TemporalDataset(
                measurements=measurements,
                descriptors=descriptors,
                obs_descriptors=obs_descriptors,
                channel_descriptors=channel_descriptors,
                time_descriptors=time_descriptors,
                check_dims=False)
            dataset_list.append(dataset)
        return dataset_list

    def split_channel(self, by):
        """ Returns a list TemporalDataset splited by channels

        Args:
            by(String): the descriptor by which the splitting is made

        Returns:
            list of TemporalDataset,
                split by the selected channel_descriptor
        """
        unique_values, inverse = get_unique_inverse(self.channel_descriptors[by])
        dataset_list = []
        for i_v, v in enumerate(unique_values):
            selection = np.where(inverse == i_v)[0]
            measurements = self.measurements[:, selection, :]
            descriptors = self.descriptors.copy()
            descriptors[by] = v
            obs_descriptors = self.obs_descriptors
            channel_descriptors = subset_descriptor(
                self.channel_descriptors, selection)
            time_descriptors = self.time_descriptors
            dataset = TemporalDataset(
                measurements=measurements,
                descriptors=descriptors,
                obs_descriptors=obs_descriptors,
                channel_descriptors=channel_descriptors,
                time_descriptors=time_descriptors,
                check_dims=False)
            dataset_list.append(dataset)
        return dataset_list

    def split_time(self, by):
        """ Returns a list TemporalDataset splited by time

        Args:
            by(String): the descriptor by which the splitting is made

        Returns:
            list of TemporalDataset,  splitted by the selected time_descriptor
        """

        time = get_unique_unsorted(self.time_descriptors[by])
        dataset_list = []
        for v in time:
            selection = [i for i, val in enumerate(self.time_descriptors[by])
                         if val == v]
            measurements = self.measurements[:, :, selection]
            descriptors = self.descriptors
            obs_descriptors = self.obs_descriptors
            channel_descriptors = self.channel_descriptors
            time_descriptors = subset_descriptor(
                self.time_descriptors, selection)
            dataset = TemporalDataset(
                measurements=measurements,
                descriptors=descriptors,
                obs_descriptors=obs_descriptors,
                channel_descriptors=channel_descriptors,
                time_descriptors=time_descriptors,
                check_dims=False)
            dataset_list.append(dataset)
        return dataset_list

    def bin_time(self, by, bins):
        """ Returns an object TemporalDataset with time-binned data.

        Args:
            bins(array-like): list of bins, with bins[i] containing the vector
                of time-points for the i-th bin

        Returns:
            a single TemporalDataset object
                Data is averaged within time-bins.
                'time' descriptor is set to the average of the
                binned time-points.
        """

        time = self.time_descriptors[by]
        n_bins = len(bins)

        binned_measurements = np.zeros((self.n_obs, self.n_channel, n_bins))
        binned_time = np.zeros(n_bins)

        for t in range(n_bins):
            t_idx = np.isin(time, bins[t])
            binned_measurements[:, :, t] = np.mean(
                self.measurements[:, :, t_idx], axis=2)
            binned_time[t] = np.mean(time[t_idx])

        time_descriptors = self.time_descriptors.copy()
        time_descriptors[by] = binned_time

        # adding the bins as an additional descriptor currently
        # does not work because of check_descriptor_length which transforms
        # it into a numpy.array.
        # time_descriptors['bins'] = [x for x in bins]
        time_descriptors['bins'] = [
            np.array2string(x, precision=2, separator=',')
            for x in bins]

        dataset = TemporalDataset(
            measurements=binned_measurements,
            descriptors=self.descriptors,
            obs_descriptors=self.obs_descriptors,
            channel_descriptors=self.channel_descriptors,
            time_descriptors=time_descriptors)
        return dataset

    def subset_obs(self, by, value):
        """ Returns a subsetted TemporalDataset defined by certain obs value

        Args:
            by(String): the descriptor by which the subset selection
                is made from obs dimension
            value:      the value by which the subset selection is made
                from obs dimension

        Returns:
            TemporalDataset, with subset defined by the selected obs_descriptor

        """
        selection = num_index(self.obs_descriptors[by], value)
        measurements = self.measurements[selection, :, :]
        descriptors = self.descriptors
        obs_descriptors = subset_descriptor(
            self.obs_descriptors, selection)
        channel_descriptors = self.channel_descriptors
        time_descriptors = self.time_descriptors
        dataset = TemporalDataset(
            measurements=measurements,
            descriptors=descriptors,
            obs_descriptors=obs_descriptors,
            channel_descriptors=channel_descriptors,
            time_descriptors=time_descriptors)
        return dataset

    def subset_channel(self, by, value):
        """ Returns a subsetted TemporalDataset defined by
        a certain channel descriptor value

        Args:
            by(String): the descriptor by which the subset selection is
                made from channel dimension
            value:      the value by which the subset selection is made
                from channel dimension

        Returns:
            TemporalDataset,
            with subset defined by the selected channel_descriptor

        """
        selection = num_index(self.channel_descriptors[by], value)
        measurements = self.measurements[:, selection]
        descriptors = self.descriptors
        obs_descriptors = self.obs_descriptors
        channel_descriptors = subset_descriptor(
            self.channel_descriptors, selection)
        time_descriptors = self.time_descriptors
        dataset = TemporalDataset(
            measurements=measurements,
            descriptors=descriptors,
            obs_descriptors=obs_descriptors,
            channel_descriptors=channel_descriptors,
            time_descriptors=time_descriptors)
        return dataset

    def subset_time(self, by, t_from, t_to):
        """ Returns a subsetted TemporalDataset
        with time between t_from and t_to

        Args:
            by(String): the descriptor by which the subset selection is
                made from channel dimension
            t_from: time-point from which onwards data should be subsetted
            t_to: time-point until which data should be subsetted

        Returns:
            TemporalDataset
                with subset defined by the selected time_descriptor

        """

        time = get_unique_unsorted(self.time_descriptors[by])
        sel_time = [t for t in time if t_from <= t <= t_to]

        selection = num_index(self.time_descriptors[by], sel_time)
        measurements = self.measurements[:, :, selection]
        descriptors = self.descriptors
        obs_descriptors = self.obs_descriptors
        channel_descriptors = self.channel_descriptors
        time_descriptors = subset_descriptor(
            self.time_descriptors, selection)
        dataset = TemporalDataset(
            measurements=measurements,
            descriptors=descriptors,
            obs_descriptors=obs_descriptors,
            channel_descriptors=channel_descriptors,
            time_descriptors=time_descriptors)
        return dataset

    def sort_by(self, by):
        """ sorts the dataset by a given observation descriptor

        Args:
            by(String): the descriptor by which the dataset shall be sorted

        Returns:
            ---

        """
        desc = self.obs_descriptors[by]
        order = np.argsort(desc)
        self.measurements = self.measurements[order]
        self.obs_descriptors = subset_descriptor(self.obs_descriptors, order)

    def convert_to_dataset(self, by):
        """ converts to Dataset long format.
            time dimension is absorbed into observation dimension

        Args:
            by(String): the descriptor which indicates the time dimension in
                the time_descriptor

        Returns:
            Dataset

        """
        time = get_unique_unsorted(self.time_descriptors[by])

        descriptors = self.descriptors
        channel_descriptors = self.channel_descriptors.copy()

        measurements = np.empty([0, self.n_channel])
        obs_descriptors = dict.fromkeys(self.obs_descriptors, [])

        for key in self.time_descriptors:
            obs_descriptors[key] = np.array([])

        for v in time:
            selection = [i for i, val in enumerate(self.time_descriptors[by])
                         if val == v]

            measurements = np.concatenate((
                measurements, self.measurements[:, :, selection].squeeze()),
                axis=0)

            for key in self.obs_descriptors:
                obs_descriptors[key] = np.concatenate((
                    obs_descriptors[key], self.obs_descriptors[key].copy()),
                    axis=0)

            for key in self.time_descriptors:
                obs_descriptors[key] = np.concatenate((
                    obs_descriptors[key], np.repeat(
                        [self.time_descriptors[key][s]
                         for s in selection],
                        self.n_obs)),
                    axis=0)

        dataset = Dataset(measurements=measurements,
                          descriptors=descriptors,
                          obs_descriptors=obs_descriptors,
                          channel_descriptors=channel_descriptors)
        return dataset

    def to_dict(self):
        """ Generates a dictionary which contains the information to
        recreate the TemporalDataset object. Used for saving to disc

        Returns:
            data_dict(dict): dictionary with TemporalDataset information

        """
        data_dict = {}
        data_dict['measurements'] = self.measurements
        data_dict['descriptors'] = self.descriptors
        data_dict['obs_descriptors'] = self.obs_descriptors
        data_dict['channel_descriptors'] = self.channel_descriptors
        data_dict['time_descriptors'] = self.channel_descriptors
        data_dict['type'] = type(self).__name__
        return data_dict


def load_dataset(filename, file_type=None):
    """ loads a Dataset object from disc

    Args:
        filename(String): path to file to load

    """
    if file_type is None:
        if isinstance(filename, str):
            if filename[-4:] == '.pkl':
                file_type = 'pkl'
            elif filename[-3:] == '.h5' or filename[-4:] == 'hdf5':
                file_type = 'hdf5'
    if file_type == 'hdf5':
        data_dict = read_dict_hdf5(filename)
    elif file_type == 'pkl':
        data_dict = read_dict_pkl(filename)
    else:
        raise ValueError('filetype not understood')
    return dataset_from_dict(data_dict)


def dataset_from_dict(data_dict):
    """ regenerates a Dataset object from the dictionary representation

    Currently this function works for Dataset, DatasetBase,
    and TemporalDataset objects

    Args:
        data_dict(dict): the dictionary representation

    Returns:
        data(Dataset): the regenerated Dataset

    """
    if data_dict['type'] == 'Dataset':
        data = Dataset(
            data_dict['measurements'],
            descriptors=data_dict['descriptors'],
            obs_descriptors=data_dict['obs_descriptors'],
            channel_descriptors=data_dict['channel_descriptors'])
    elif data_dict['type'] == 'DatasetBase':
        data = DatasetBase(
            data_dict['measurements'],
            descriptors=data_dict['descriptors'],
            obs_descriptors=data_dict['obs_descriptors'],
            channel_descriptors=data_dict['channel_descriptors'])
    elif data_dict['type'] == 'TemporalDataset':
        data = TemporalDataset(
            data_dict['measurements'],
            descriptors=data_dict['descriptors'],
            obs_descriptors=data_dict['obs_descriptors'],
            channel_descriptors=data_dict['channel_descriptors'],
            time_descriptors=data_dict['time_descriptors'])
    else:
        raise ValueError('type of Dataset not recognized')
    return data


def merge_subsets(dataset_list):
    """
    Generate a dataset object from a list of smaller dataset objects
    (e.g., as generated by the subset_* methods). Assumes that descriptors,
    channel descriptors and number of channels per observation match.

    Args:
        dataset_list (list):
            List containing rsatoolbox datasets

    Returns:
        merged_dataset (Dataset):
            rsatoolbox dataset created from all datasets in dataset_list
    """
    assert isinstance(dataset_list, list), "Provided object is not a list."
    assert "Dataset" in str(type(dataset_list[0])), \
        "Provided list does not only contain Dataset objects."
    baseline_ds = dataset_list[0]
    descriptors = baseline_ds.descriptors.copy()
    channel_descriptors = baseline_ds.channel_descriptors.copy()
    measurements = baseline_ds.measurements.copy()
    obs_descriptors = baseline_ds.obs_descriptors.copy()

    for ds in dataset_list[1:]:
        assert "Dataset" in str(type(ds)), \
            "Provided list does not only contain Dataset objects."
        assert descriptors == ds.descriptors.copy(), \
            "Dataset descriptors do not match."
        measurements = np.append(measurements, ds.measurements, axis=0)
        obs_descriptors = append_obs_descriptors(obs_descriptors,
                                                 ds.obs_descriptors.copy())

    merged_dataset = Dataset(measurements,
                             descriptors=descriptors,
                             obs_descriptors=obs_descriptors,
                             channel_descriptors=channel_descriptors)
    return merged_dataset
