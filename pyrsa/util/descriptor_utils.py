#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Descriptor handling

@author: heiko
"""

import numpy as np


def bool_index(descriptor, value):
    """
    creates a boolean index vector where a descriptor has a value

        Args:
            descriptor(np.ndarray): descriptor vector
            value:                  value or list of values to mark

        Returns:
            bool_index:         boolean index vector where descriptor == value
    """
    if (type(value) is list or
            type(value) is tuple or
            type(value) is np.ndarray):
        bool_index = np.array([descriptor == v for v in value])
        bool_index = np.any(bool_index, axis=0)
    else:
        bool_index = np.array(descriptor == value)
    return bool_index


def format_descriptor(descriptors):
    """ formats a descriptor dictionary
        Args:
            descriptors(dict): the descriptor dictionary

        Returns:
            string_descriptors(String): formated string to show dict
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
            descriptors(dict): descriptor dictionary
    """
    if descriptors is None:
        descriptors = {}
    elif not isinstance(descriptors, dict):
        raise ValueError('Descriptors must be dictionaries!')
    return descriptors


def check_descriptor_length(descriptor, n):
    """
    Checks whether the entries of a descriptor dictionary have the right length

        Args:
            descriptor(dict): the descriptor dictionary
            n: the correct length of the descriptors

        Returns:
            bool
    """
    for k, v in descriptor.items():
        if np.array(v).shape[0] != n:
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
    return extracted_descriptor


def check_descriptor_length_error(descriptor, name, n):
    """
    Raises an error if the given descriptor does not have the right length

        Args:
            descriptor(dict/None): the descriptor dictionary
            name(String): Descriptor name used for error message
            n: the indices to be extracted

        Returns:
            ---
    """
    if descriptor is not None:
        if not check_descriptor_length(descriptor, n):
            raise AttributeError(
                name + " have mismatched dimension with measurements.")
