#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Definition of RSA RDMs class and subclasses
@author: baihan
"""

import numpy as np
from scipy.spatial.distance import squareform
# import pyrsa as rsa

class RDMs:
    """
    RDMs class.

        Args:
            dissimilarities (numpy.ndarray):
                either a 2d np-array (n_rdm x vectorform of dissimilarities)
                or a 3d np-array (n_rdm x n_cond x n_cond)
            dissimilarity_measure (String):     a description of the dissimilarity measure
                (e.g. 'Euclidean')      
            descriptors (dict):     descriptors with 1 value per RDMs object
        Returns:
            RDMs object
    """
    def __init__(self, dissimilarities=None, dissimilarity_measure=None, descriptors=None):
        if (dissimilarities.ndim == 2):
            self.dissimilarities = dissimilarities
            self.n_rdm = self.measurements.shape[0]
            self.n_cond = np.ceil(np.sqrt(self.measurements.shape[0]*2))
        elif (dissimilarities.ndim == 3):
            self.dissimilarities = dissimilarities
            self.n_rdm = self.dissimilarities.shape[0]
            self.n_cond = self.dissimilarities.shape[1]
        self.descriptors = descriptors 
        self.dissimilarity_measure = dissimilarity_measure 

    def get_vectors(self):
        """ Returns RDMs as np.ndarray with each RDM as a vector
        Returns:
            RDMs as with one vector as one RDM
        """
        return self.dissimilarities

    def get_matrices(self):
        """ Returns RDMs as np.ndarray with each RDM as a matrix
        Returns:
            RDMs as with one matrix as one RDM
        """
        RDMs_matrix = np.ndarray((n_rdm, n_cond, n_cond))
        for idx in np.arange(n_rdm):
            RDMs_matrix[idx, :, :] = squareform(self.dissimilarities[idx])
        return RDMs_matrix
