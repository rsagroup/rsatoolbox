#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
saving to and reading from files
"""

import os
from collections.abc import Iterable
from pathlib import Path
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
    file.attrs['rsatoolbox_version'] = '0.0.1'
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
        elif isinstance(value, list):
            _write_list(group, key, value)
        elif isinstance(value, dict):
            subgroup = group.create_group(key)
            _write_to_group(subgroup, value)
        elif value is None:
            group[key] = h5py.Empty("f")
        elif isinstance(value, Iterable):
            if isinstance(value[0], str):
                group.attrs[key] = value
        else:
            group[key] = value


def _write_list(group, key, value):
    """
    writes a list to a hdf5 file. First tries conversion to np.array.
    If this fails the list is converted to a dict with integer keys.

    Parameters
    ----------
    group : hdf5 group
        where to write.
    key :  hdf5 key
    value : list
        list to be written
    """
    try:
        value = np.array(value)
        if str(value.dtype)[:2] == '<U':
            group[key] = value.astype('S')
        else:
            group[key] = value
    except TypeError:
        l_group = group.create_group(key)
        for i, v in enumerate(value):
            l_group[str(i)] = v


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
    dictionary['rsatoolbox_version'] = '0.0.1'
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


def remove_file(file):
    """ Deletes file from OS if it exists

    Args:
        file (str, Path):
            a filename or opened readable file

    """
    if isinstance(file, (str, Path)) and os.path.exists(file):
        os.remove(file)
    elif hasattr(file, 'name') and os.path.exists(file.name):
        file.truncate(0)
