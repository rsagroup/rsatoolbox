#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
saving to and reading from files
"""

import os
import pickle
import h5py
import numpy as np


def write_dict_hdf5(file, dictionary):
    """ writes a nested dictionary containing strings & arrays as data into
    a hdf5 file

    Args:
        file: a filename or opened writable file
        dictionary(dict): the dict to be saved

    """
    if isinstance(file, str):
        if os.path.exists(file):
            raise ValueError('File already exists!')
    file = h5py.File(file, 'a')
    file.attrs['pyrsa_version'] = '3.0'
    _write_to_group(file, dictionary)


def _write_to_group(group, dictionary):
    """ writes a dictionary to a hdf5 group, which can recurse"""
    for key in dictionary.keys():
        value = dictionary[key]
        if isinstance(value, str):
            # needs another conversion to string to catch weird subtypes
            # like numpy.str_
            group.attrs[key] = str(value)
        elif isinstance(value, np.ndarray):
            if str(value.dtype)[:2] == '<U':
                group[key] = value.astype('S')
            else:
                group[key] = value
        elif isinstance(value, dict):
            subgroup = group.create_group(key)
            _write_to_group(subgroup, value)
        elif value is None:
            group[key] = h5py.Empty("f")
        else:
            group[key] = value


def read_dict_hdf5(file):
    """ writes a nested dictionary containing strings & arrays as data into
    a hdf5 file

    Args:
        file: a filename or opened readable file

    Returns:
        dictionary(dict): the loaded dict

    """
    file = h5py.File(file, 'r')
    return _read_group(file)


def _read_group(group):
    """ reads a group from a hdf5 file into a dict, which allows recursion"""
    dictionary = {}
    for key in group.keys():
        if isinstance(group[key], h5py.Group):
            dictionary[key] = _read_group(group[key])
        elif group[key].shape is None:
            dictionary[key] = None
        else:
            dictionary[key] = np.array(group[key])
            if dictionary[key].dtype.type is np.string_:
                dictionary[key] = np.array(group[key]).astype('unicode')
            # if (len(dictionary[key].shape) == 1
            #     and dictionary[key].shape[0] == 1):
            #     dictionary[key] = dictionary[key][0]
    for key in group.attrs.keys():
        dictionary[key] = group.attrs[key]
    return dictionary


def write_dict_pkl(file, dictionary):
    """ writes a nested dictionary containing strings & arrays as data into
    a pickle file

    Args:
        file: a filename or opened writable file
        dictionary(dict): the dict to be saved

    """
    if isinstance(file, str):
        file = open(file, 'wb')
    dictionary['pyrsa_version'] = '3.0'
    pickle.dump(dictionary, file, protocol=-1)


def read_dict_pkl(file):
    """ writes a nested dictionary containing strings & arrays as data into
    a pickle file

    Args:
        file: a filename or opened readable file

    Returns:
        dictionary(dict): the loaded dict


    """
    if isinstance(file, str):
        file = open(file, 'rb')
    data = pickle.load(file)
    return data
