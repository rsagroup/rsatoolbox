#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Definition of RSA RDMs class and subclasses
@author: baihan
"""

import numpy as np
from pyrsa.util.rdm_utils import batch_to_vectors
from pyrsa.util.rdm_utils import batch_to_matrices


class RDMs:
    """
    RDMs class.

        Args:
            dissimilarities (numpy.ndarray):
                either a 2d np-array (n_rdm x vectorform of dissimilarities)
                or a 3d np-array (n_rdm x n_cond x n_cond)
            dissimilarity_measure (String):
                a description of the dissimilarity measure (e.g. 'Euclidean')
            descriptors (dict):
                descriptors with 1 value per RDMs object
        Returns:
            RDMs object
    """
    def __init__(self, dissimilarities=None,
                 dissimilarity_measure=None,
                 descriptors=None):
        self.dissimilarities, self.n_rdm, self.n_cond = \
            batch_to_vectors(dissimilarities)
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
        matrices, _, _ = batch_to_matrices(self.dissimilarities)
        return matrices
