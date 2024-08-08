#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Descriptor handling.
Note: descriptor is assumed to be a list, to accommodate objects that don't fit well into strings,
such as arrays of varying sizes.
Some of these methods may convert numpy-array descriptors to list-types.

@author: adkipnis
"""
from __future__ import annotations
from typing import TYPE_CHECKING, Dict, List
from collections.abc import Iterable
import numpy as np
if TYPE_CHECKING:
    DescriptorDict = Dict[str, List | np.ndarray]


def bool_index(descriptor, value):
    """
    creates a boolean index vector where a descriptor has a value

    Args:
        descriptor (list-like): descriptor vector
        value:                  value or list of values to mark

    Returns:
        numpy.ndarray:
            bool_index: boolean index vector where descriptor == value

    """
    descriptor = np.array(descriptor)
    if isinstance(value, (list, tuple, np.ndarray)):
        index = np.array([descriptor == v for v in value])
        index = np.any(index, axis=0)
    else:
        index = np.array(descriptor == value)
    return index


def num_index(descriptor, value):
    """
    creates a boolean index vector where a descriptor has a value

    Args:
        descriptor (list-like): descriptor vector
        value:                  value or list of values to mark

    Returns:
        numpy.ndarray:
            bool_index: boolean index vector where descriptor == value

    """
    return np.where(bool_index(descriptor, value))[0]


def format_descriptor(descriptors):
    """ formats a descriptor dictionary

    Args:
        descriptors(dict): the descriptor dictionary

    Returns:
        String: formatted string to show dict

    """
    string_descriptors = ''
    for entry in descriptors:
        string_descriptors = (string_descriptors +
                              f'{entry} = {descriptors[entry]}\n'
                              )
    return string_descriptors


def parse_input_descriptor(descriptors):
    """ parse input descriptor checks whether an input descriptors dictionary
    is a dictionary. If it is None instead it is replaced by an empty dict.
    Otherwise an error is raised.

    Args:
        descriptors(dict/None): the descriptor dictionary

    Returns:
        dict: descriptor dictionary

    """
    if descriptors is None:
        descriptors = {}
    elif not isinstance(descriptors, dict):
        raise ValueError('Descriptors must be dictionaries!')
    return descriptors


def check_descriptor_length(descriptor, n_element):
    """
    Checks whether the entries of a descriptor dictionary have the right length.
    Converts single-strings to a list of 1 element.

    Args:
        descriptor(dict): the descriptor dictionary
        n_element: the correct length of the descriptors

    Returns:
        bool

    """
    for k, v in descriptor.items():
        if isinstance(v, str):
            v = [v]
        if isinstance(v, Iterable) and len(v) != n_element:
            return False
    return True


def subset_descriptor(descriptor, indices):
    """
    Retrieves a subset of a descriptor given by indices.

    Args:
        descriptor(dict): the descriptor dictionary
        indices: the indices to be extracted

    Returns:
        extracted_descriptor(dict): the selected subset of the descriptor

    """
    extracted_descriptor = {}
    if isinstance(indices, Iterable):
        for k, v in descriptor.items():
            extracted_descriptor[k] = [v[index] for index in indices]
    else:
        for k, v in descriptor.items():
            extracted_descriptor[k] = [v[indices]]
    return extracted_descriptor


def append_descriptor(descriptor, desc_new):
    """
    appends a descriptor to an existing one

    Args:
        descriptor(dict): the descriptor dictionary, with list-like values
        desc_new(dict): the descriptor dictionary to append

    Returns:
        descriptor(dict): the longer descriptor

    """
    for k, v in descriptor.items():
        assert k in desc_new.keys(), f'appended descriptors misses key {k}'
        descriptor[k] = list(v) + list(desc_new[k])
    descriptor['index'] = list(range(len(descriptor['index'])))
    return descriptor


def check_descriptor_length_error(descriptor, name, n_element):
    """
    Raises an error if the given descriptor does not have the right length

    Args:
        descriptor(dict/None): the descriptor dictionary
        name(String): Descriptor name used for error message
        n_element: the desired descriptor length

    Returns:
        ---

    """
    if descriptor is not None:
        if not check_descriptor_length(descriptor, n_element):
            raise AttributeError(
                name + " have mismatched dimension with measurements.")


def dict_to_list(d_dict):
    """
    converts a dictionary from a hdf5 file to a list
    """
    for k, v in d_dict.items():
        if isinstance(v, dict):
            d_dict[k] = [
                d_dict[k][str(i)]
                for i in range(len(d_dict[k]))]
        else:
            d_dict[k] = list(d_dict[k])
    return d_dict


def desc_eq(a: DescriptorDict, b: DescriptorDict) -> bool:
    """Whether the two descriptor-style dictionaries are equal

    Args:
        a (dict): Dictionary with list or array values
        b (dict): Dictionary with list or array values

    Returns:
        bool: True if the two dictionaries have the same keys and values
    """
    if set(a.keys()) == set(b.keys()):
        return all(
            all(np.asarray(a[k]) == np.asarray(b[k])) for k in a.keys()
        )
    return False
