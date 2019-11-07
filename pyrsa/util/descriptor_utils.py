#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Descriptor handling

@author: heiko
"""


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
