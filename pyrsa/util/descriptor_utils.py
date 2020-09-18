#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Descriptor handling
"""

import numpy as np


def bool_index(descriptor, value):
    """
    creates a boolean index vector where a descriptor has a value

    Args:
        descriptor(numpy.ndarray): descriptor vector
        value:                  value or list of values to mark

    Returns:
        numpy.ndarray:
            bool_index: boolean index vector where descriptor == value

    """
    descriptor = np.array(descriptor)
    if (type(value) is list or
            type(value) is tuple or
            type(value) is np.ndarray):
        index = np.array([descriptor == v for v in value])
        index = np.any(index, axis=0)
    else:
        index = np.array(descriptor == value)
    return index


def format_descriptor(descriptors):
    """ formats a descriptor dictionary

    Args:
        descriptors(dict): the descriptor dictionary

    Returns:
        String: formated string to show dict

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
    Checks whether the entries of a descriptor dictionary have the right length

    Args:
        descriptor(dict): the descriptor dictionary
        n_element: the correct length of the descriptors

    Returns:
        bool

    """
    for k, v in descriptor.items():
        v = np.asarray(v)
        if not v.shape:
            # 0-d array happens e.g. when casting str to array
            v = v.flatten()
        descriptor[k] = v
        if v.shape[0] != n_element:
            return False
    return True


def subset_descriptor(descriptor, indices):
    """
    retrievs a subset of a descriptor given by indices.

    Args:
        descriptor(dict): the descriptor dictionary
        indices: the indices to be extracted

    Returns:
        extracted_descriptor(dict): the selected subset of the descriptor

    """
    extracted_descriptor = {}
    for k, v in descriptor.items():
        if isinstance(indices, tuple) or isinstance(indices, list):
            extracted_descriptor[k] = [v[index] for index in indices]
        else:
            extracted_descriptor[k] = np.array(v)[indices]
        if len(np.array(extracted_descriptor[k]).shape) == 0:
            extracted_descriptor[k] = [extracted_descriptor[k]]
    return extracted_descriptor


def append_descriptor(descriptor, desc_new):
    """
    appends a descriptor to an existing one

    Args:
        descriptor(dict): the descriptor dictionary
        desc_new(dict): the descriptor dictionary to append

    Returns:
        descriptor(dict): the longer descriptor

    """
    for k, v in descriptor.items():
        assert k in desc_new.keys(), f'appended descriptors misses key {k}'
        d_new = np.array(desc_new[k])
        v = np.array(v)
        if not v.shape:
            v = v.flatten()
        if not d_new.shape:
            descriptor[k] = np.concatenate((v, d_new.flatten()), axis=0)
        else:
            descriptor[k] = np.concatenate((v, d_new), axis=0)
    descriptor['index'] = np.arange(len(descriptor['index']))
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
