#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
matlab_io
Utility functions for loading matlab data
@author: johan
"""

import os
import os.path
import scipy.io
from pyrsa import vis

DEMO_DIR = os.path.dirname(os.path.realpath("__file__"))

NEURON_DIR = os.path.join(DEMO_DIR, "92imageData")


def neuron_2008_images():
    """ Load Krigeskorte et al. (2008, Neuron) images as Icon instances."""
    mat_path = os.path.join(NEURON_DIR, "Kriegeskorte_Neuron2008_supplementalData.mat")
    mat = scipy.io.loadmat(mat_path)
    return [
        vis.Icon(image=this_image) for this_image in mat["stimuli_92objs"][0]["image"]
    ]
