#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 16:21:53 2020

@author: heiko
"""

import numpy as np
import matplotlib.pyplot as plt


def show_rdm(rdm, do_rank_transform=False):
    """shows an rdm object

    Parameters
    ----------
    rdm : pyrsa.rdm.RDMs
        RDMs object to be plotted

    """
    rdm_mat = rdm.get_matrices()
    if do_rank_transform:
        rdm_mat = rank_transform(rdm_mat)
    if rdm.n_rdm  > 1:
        m = np.ceil(np.sqrt(rdm.n_rdm+1))
        n = np.ceil((1 + rdm.n_rdm) / m)
        for idx in range(rdm.n_rdm):
            plt.subplot(n, m, idx)
            plt.imshow(rdm_mat[idx])
        plt.subplot(n, m, rdm.n_rdm)
        plt.imshow(np.mean(rdm_mat, axis=0))
    elif rdm.n_rdm == 1:
        plt.imshow(rdm_mat[0])
    plt.show()
