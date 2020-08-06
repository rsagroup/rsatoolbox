#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Definition of RSA Dataset class and subclasses
@author: baihan, jdiedrichsen, adkipnis
"""


import numpy as np
import pickle
from pyrsa.util.data_utils import get_unique_unsorted
from pyrsa.util.descriptor_utils import check_descriptor_length_error
from pyrsa.util.descriptor_utils import subset_descriptor
from pyrsa.util.descriptor_utils import bool_index
from pyrsa.util.descriptor_utils import format_descriptor
from pyrsa.util.descriptor_utils import parse_input_descriptor
from pyrsa.util.file_io import write_dict_hdf5
from pyrsa.util.file_io import write_dict_pkl
from pyrsa.util.file_io import read_dict_hdf5
from pyrsa.util.file_io import read_dict_pkl


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
                 obs_descriptors=None, channel_descriptors=None):
        if measurements.ndim != 2:
            raise AttributeError(
                "measurements must be in dimension n_obs x n_channel")
        self.measurements = measurements
        self.n_obs, self.n_channel = self.measurements.shape
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
        return (f'pyrsa.data.{self.__class__.__name__}(\n'
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
        return (f'pyrsa.data.{self.__class__.__name__}\n'
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

    def save(self, filename, file_type='hdf5'):
        """ Saves the dataset object to a file

        Args:
            filename(String): path to the file
                [or opened file]
            file_type(String): Type of file to create:
                hdf5: hdf5 file
                pkl: pickle file

        """
        data_dict = self.to_dict()
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
            obs_descriptors = subset_descriptor(
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
            descriptors[by] = v
            obs_descriptors = self.obs_descriptors
            channel_descriptors = subset_descriptor(
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
        selection = bool_index(self.obs_descriptors[by], value)
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
        selection = bool_index(self.channel_descriptors[by], value)
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

    Currently this function works for Dataset and DatasetBase objects

    Args:
        data_dict(dict): the dictionary representation

    Returns:
        data(Dataset): the regenerated Dataset

    """
    if data_dict['type'] == 'Dataset':
        data = Dataset(data_dict['measurements'],
                       descriptors=data_dict['descriptors'],
                       obs_descriptors=data_dict['obs_descriptors'],
                       channel_descriptors=data_dict['channel_descriptors'])
    elif data_dict['type'] == 'DatasetBase':
        data = DatasetBase(
            data_dict['measurements'],
            descriptors=data_dict['descriptors'],
            obs_descriptors=data_dict['obs_descriptors'],
            channel_descriptors=data_dict['channel_descriptors'])
    else:
        raise ValueError('type of Dataset not recognized')
    return data


def merge_subsets(dataset_list):
    """
    Generate a dataset object from a list of smaller dataset objects (e.g., as generated by the subset_* methods).
    Assumes that descriptors, channel descriptors and number of channels per observation match.

    Args:
        dataset_list (list):
            List containing PyRSA datasets
    
    Returns:
        merged_dataset (Dataset):
            PyRSA dataset created from all datasets in dataset_list
    """
    assert isinstance(dataset_list, list), "Provided object is not a list."
    assert "Dataset" in str(type(dataset_list[0])), "Provided list does not only contain Dataset objects."
    
    baseline_ds = dataset_list[0]
    descriptors = baseline_ds.descriptors.copy()
    channel_descriptors = baseline_ds.channel_descriptors.copy()
    measurements = baseline_ds.measurements.copy()
    obs_descriptors = baseline_ds.obs_descriptors.copy()
    
    for ds in dataset_list[1:]:
        assert "Dataset" in str(type(ds)), "Provided list does not only contain Dataset objects."
        assert descriptors == ds.descriptors.copy(), "Dataset descriptors do not match."
        measurements = np.append(measurements, ds.measurements, axis=0)
        obs_descriptors = append_obs_descriptors(obs_descriptors, ds.obs_descriptors.copy())
    
    merged_dataset = Dataset(measurements,
                                descriptors = descriptors, 
                                obs_descriptors = obs_descriptors,                             
                                channel_descriptors = channel_descriptors)
    return merged_dataset


def append_obs_descriptors(dict_orig, dict_addit):
    """
    Merge two dictionaries of observation descriptors with matching keys and numpy arrays as values.
    """
    assert list(dict_orig.keys()) == list(dict_addit.keys()), "Provided observation descriptors have different keys."
    
    dict_merged = {}
    keys = list(dict_orig.keys())
    for k in keys:
        values = np.array(np.append(dict_orig[k], dict_addit[k]))
        dict_merged.update({k:values})
    return dict_merged

def odd_even_split(dataset, obs_desc, sort_by = None):
    """
    Perform a simple odd-even split on a PyRSA dataset. It will be partitioned into n different datasets,
    where n is the number of distinct values on dataset.obs_descriptors[obs_desc]. The resulting list will be split into odd and even (index) subset.
    The datasets contained in these subsets will then be merged.
    
    Args:
        dataset (Dataset):
            PyRSA dataset which will be split
        obs_desc (str):
            Observation descriptor, basis for partitioning (must contained in keys of dataset.obs_descriptors)
        sort_by (str or None):
            If str, the resulting splits will be sorted according to sort_by (must be contained in keys of dataset.obs_descriptors)
    
    Returns:
        odd_split (Dataset):
            subset of the Dataset with odd list-indices after partitioning according to obs_desc
        even_split (Dataset):
            subset of the Dataset with even list-indices after partitioning according to obs_desc
    """
    assert obs_desc in dataset.obs_descriptors.keys(), "obs_desc must be contained in keys of dataset.obs_descriptors"
    if isinstance(sort_by, str):
        assert sort_by in dataset.obs_descriptors.keys(), "sort_by must be contained in keys of dataset.obs_descriptors"
        
    ds_part = dataset.split_obs(obs_desc)
    odd_list = ds_part[0::2]
    even_list = ds_part[1::2]
    odd_split = merge_subsets(odd_list)
    even_split = merge_subsets(even_list)
    
    if sort_by is not None:
        odd_split.sort_by(sort_by)
        even_split.sort_by(sort_by)
    return odd_split, even_split


def nested_odd_even_split(dataset, l1_obs_desc, l2_obs_desc, sort_by = None):
    """
    Nested version of odd_even_split, where dataset is first partitioned according to the l1_obs_desc
    and each partition is again partitioned according to the l2_obs_desc (after which the actual oe-split occurs).
    
    Args:
        dataset (Dataset):
            PyRSA dataset which will be split
        l1_obs_desc (str):
            Observation descriptor, basis for level 1 partitioning (must contained in keys of dataset.obs_descriptors)
        sort_by (str or None):
            If str, the resulting splits will be sorted according to sort_by (must be contained in keys of dataset.obs_descriptors)
    
    Returns:
        odd_split (Dataset):
            subset of the Dataset with odd list-indices after partitioning according to obs_desc
        even_split (Dataset):
            subset of the Dataset with even list-indices after partitioning according to obs_desc

    """
    assert l1_obs_desc and l2_obs_desc in dataset.obs_descriptors.keys(), "obs_desc must be contained in keys of dataset.obs_descriptors"
    if isinstance(sort_by, str):
        assert sort_by in dataset.obs_descriptors.keys(), "sort_by must be contained in keys of dataset.obs_descriptors"
        
    ds_part = dataset.split_obs(l1_obs_desc)
    odd_list = []
    even_list = []
    for partition in ds_part:
        odd_split, even_split = odd_even_split(partition, l2_obs_desc)
        odd_list.append(odd_split)
        even_list.append(even_split)
    odd_split = merge_subsets(odd_list)
    even_split = merge_subsets(even_list)
    if sort_by is not None:
        odd_split.sort_by(sort_by)
        even_split.sort_by(sort_by)
    return odd_split, even_split