#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Definition of RSA Dataset classes for neuroimaging or neural data
@author: baihan
"""


from pyrsa.data.dataset import DatasetBase


def preprocess_fmri(rawdata=None):
    """example of a preprocessing function

        Args:
            rawdata (pyrsa.data.dataset.Dataset): the neural data

        Returns:
            preprocessed neural data in format of measurements,
            descriptors, obs_descriptors, channel_descriptors

        Example:
            .. code-block:: python

                measurements, descriptors, obs_descriptors,
                              channel_descriptors = preprocess(rawdata)

    """
    raise NotImplementedError(
        "preprocess_fmri function not implemented!")
