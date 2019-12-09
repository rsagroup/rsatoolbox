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
from pyrsa.util.descriptor_utils import bool_index
from pyrsa.util.data_utils import extract_dict

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
            pattern_descriptors (dict)
                descriptors with 1 value per RDM column
        Returns:
            RDMs object
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

    def subset_patterns(self, by, value):
        """ Returns a smaller RDMs with patterns with certain descriptor values
        Args:
            by(String): the descriptor by which the subset selection
                        is made from pattern_descriptors
            value:      the value by which the subset selection is made
                        from pattern_descriptors

        Returns:
            RDMs object, with fewer patterns
        """
        selection = bool_index(self.pattern_descriptors[by], value)
        dissimilarities = self.get_matrices()[:,selection, selection]
        descriptors = self.descriptors
        pattern_descriptors = extract_dict(
            self.pattern_descriptors, selection)
        rdms = RDMs(dissimilarities=batch_to_vectors(dissimilarities),
                    descriptors=descriptors,
                    pattern_descriptors=pattern_descriptors)
        return rdms

    def subset(self, by, value):
        """ Returns a set of fewer RDMs matching descriptor values
        Args:
            by(String): the descriptor by which the subset selection
                        is made from descriptors
            value:      the value by which the subset selection is made
                        from descriptors

        Returns:
            RDMs object, with fewer RDMs
        """
        selection = bool_index(self.descriptors[by], value)
        dissimilarities = self.dissimilarities[selection, :]
        pattern_descriptors = self.pattern_descriptors
        descriptors = extract_dict(
            self.descriptors, selection)
        rdms = RDMs(dissimilarities=dissimilarities,
                    descriptors=descriptors,
                    pattern_descriptors=pattern_descriptors)
        return rdms
