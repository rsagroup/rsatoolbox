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
from pyrsa import rdm

DEMO_DIR = os.path.dirname(os.path.realpath("__file__"))

NEURON_DIR = os.path.join(DEMO_DIR, "92imageData")


def neuron_2008_images():
    """ Load Krigeskorte et al. (2008, Neuron) images as Icon instances."""
    mat_path = os.path.join(NEURON_DIR, "Kriegeskorte_Neuron2008_supplementalData.mat")
    mat = scipy.io.loadmat(mat_path)
    return [
        vis.Icon(image=this_image) for this_image in mat["stimuli_92objs"][0]["image"]
    ]


def neuron_2008_rdms_fmri():
    """ Load Kriegeskorte et al. (2008, Neuron) fMRI RDMs as RDMs instance."""
    mat_path = os.path.join(NEURON_DIR, "92_brainRDMs.mat")
    mat = scipy.io.loadmat(mat_path)
    # insert leading dim to conform with pyrsa nrdm x ncon x ncon convention
    return rdm.concat(
        [
            rdm.RDMs(
                dissimilarities=this_rdm["RDM"][None, :, :],
                dissimilarity_measure="pearson",
                rdm_descriptors=dict(
                    zip(["ROI", "subject", "session"], this_rdm["name"][0].split(" | "))
                ),
            )
            for this_rdm in mat["RDMs"].flatten()
        ]
    )
