#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: baihan
"""

import numpy as np
import matplotlib.pyplot as plt
import pyrsa.vis as rsv

def plot_mds(rdms,frame):
    """ plots multi-dimensional scaling of RDMs class

    Args:
        rdms (RDMs class): an RDMs class object
        frame (int): the frame number to be plotted

    Returns:
        ---

    """
    mds_emb = rsv.vis.mds(rdms)
    plt.figure(figsize=(10, 10))
    plt.scatter(mds_emb[frame,:,0],mds_emb[i,:,1])
