#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 14:04:52 2020

@author: heiko
"""

import matplotlib.pyplot as plt


def plot_model_comparison(evaluations):
    """ Returns the predicted rdm vector

    theta are the weights for the different rdms

    Args:
        theta(numpy.ndarray): the model parameter vector (one dimensional)

    Returns:
        rdm vector

    """

