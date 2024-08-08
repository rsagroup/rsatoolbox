"""
saving to and reading from HDF5 files
"""
from __future__ import annotations
from typing import Union, Dict, List, IO
import os
from collections.abc import Iterable
try:  # drop:py37 (backport)
    from importlib.metadata import version
except ModuleNotFoundError:
    from importlib_metadata import version
from h5py import File, Group, Empty
import numpy as np


def write_dict_hdf5(fhandle: Union[str, IO], dictionary: Dict) -> None:
    """ writes a nested dictionary containing strings & arrays as data into
    a hdf5 file

    Args:
        file: a filename or opened writable file
        dictionary(dict): the dict to be saved

    """
    if isinstance(fhandle, str):
        if os.path.exists(fhandle):
            raise ValueError('File already exists!')
    file = File(fhandle, 'a')
    file.attrs['rsatoolbox_version'] = version('rsatoolbox')
    _write_to_group(file, dictionary)


def _write_to_group(group: Group, dictionary: Dict) -> None:
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
            group[key] = Empty("f")
        elif isinstance(value, Iterable):
            if isinstance(value[0], str):
                group.attrs[key] = value
        else:
            group[key] = value


def _write_list(group: Group, key: str, value: List) -> None:
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


def read_dict_hdf5(fhandle: Union[str, IO]) -> Dict:
    """ writes a nested dictionary containing strings & arrays as data into
    a hdf5 file

    Args:
        file: a filename or opened readable file

    Returns:
        dictionary(dict): the loaded dict

    """
    file = File(fhandle, 'r')
    return _read_group(file)


def _read_group(group: Group) -> Dict:
    """ reads a group from a hdf5 file into a dict, which allows recursion"""
    dictionary = {}
    for key in group.keys():
        sub_val = group[key]
        if isinstance(sub_val, Group):
            dictionary[key] = _read_group(sub_val)
        elif sub_val.shape is None:
            dictionary[key] = None
        else:
            dictionary[key] = np.array(sub_val)
            if dictionary[key].dtype.type is np.bytes_:
                dictionary[key] = np.array(sub_val).astype('unicode')
            # if (len(dictionary[key].shape) == 1
            #     and dictionary[key].shape[0] == 1):
            #     dictionary[key] = dictionary[key][0]
    for key in group.attrs.keys():
        dictionary[key] = group.attrs[key]
    return dictionary
