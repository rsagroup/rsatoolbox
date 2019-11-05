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