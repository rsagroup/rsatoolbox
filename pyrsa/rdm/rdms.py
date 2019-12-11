#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Definition of RSA RDMs class and subclasses
@author: baihan
"""

import numpy as np
from pyrsa.util.rdm_utils import batch_to_vectors
from pyrsa.util.rdm_utils import batch_to_matrices
from pyrsa.util.descriptor_utils import format_descriptor


class RDMs:
    """ RDMs class

    Args:
        dissimilarities (numpy.ndarray):
            either a 2d np-array (n_rdm x vectorform of dissimilarities)
            or a 3d np-array (n_rdm x n_cond x n_cond)
        dissimilarity_measure (String):
            a description of the dissimilarity measure (e.g. 'Euclidean')
        descriptors (dict):
            descriptors with 1 value per RDMs object
        pattern_descriptors (dict):
            descriptors with 1 value per RDM column

    Attributes:
        n_rdm(int): number of rdms
        n_cond(int): number of patterns

    """
    def __init__(self, dissimilarities,
                 dissimilarity_measure=None,
                 descriptors={},
                 pattern_descriptors={}):
        self.dissimilarities, self.n_rdm, self.n_cond = \
            batch_to_vectors(dissimilarities)
        if descriptors is None:
            self.descriptors = {}
        else:
            self.descriptors = descriptors
        self.dissimilarity_measure = dissimilarity_measure
        self.pattern_descriptors = pattern_descriptors

    def __repr__(self):
        """
        defines string which is printed for the object
        """
        return (f'pyrsa.rdm.{self.__class__.__name__}(\n'
                f'dissimilarity_measure = \n{self.dissimilarity_measure}\n'
                f'dissimilarities = \n{self.dissimilarities}\n'
                f'descriptors = \n{self.descriptors}\n'
                )

    def __str__(self):
        """
        defines the output of print
        """
        string_desc = format_descriptor(self.descriptors)
        diss = self.get_matrices()[0]
        return (f'pyrsa.rdm.{self.__class__.__name__}\n'
                f'{self.n_rdm} RDM(s) over {self.n_cond} conditions\n\n'
                f'dissimilarity_measure = \n{self.dissimilarity_measure}\n\n'
                f'dissimilarities[0] = \n{diss}\n\n'
                f'descriptors: \n{string_desc}\n'
                )

    def get_vectors(self):
        """ Returns RDMs as np.ndarray with each RDM as a vector
        
        Returns:
            numpy.ndarray: RDMs as with one vector per RDM

        """
        return self.dissimilarities

    def get_matrices(self):
        """ Returns RDMs as np.ndarray with each RDM as a matrix

        Returns:
            numpy.ndarray: RDMs as with one matrix per RDM

        """
        matrices, _, _ = batch_to_matrices(self.dissimilarities)
        return matrices
