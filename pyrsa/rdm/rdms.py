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
from pyrsa.util.data_utils import check_descriptors_dimension


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
        rdm_descriptors (dict):
            descriptors with 1 value per RDM
        pattern_descriptors (dict):
            descriptors with 1 value per RDM column

    Attributes:
        n_rdm(int): number of rdms
        n_cond(int): number of patterns

    """
    def __init__(self, dissimilarities,
                 dissimilarity_measure=None,
                 descriptors=None,
                 rdm_descriptors=None,
                 pattern_descriptors=None):
        self.dissimilarities, self.n_rdm, self.n_cond = \
            batch_to_vectors(dissimilarities)
        if descriptors is None:
            self.descriptors = {}
        else:
            self.descriptors = descriptors
        if rdm_descriptors is None:
            self.rdm_descriptors = {}
        else:
            check_descriptors_dimension(rdm_descriptors,
                                        'rdm_descriptors',
                                        self.n_rdm)
            self.rdm_descriptors = rdm_descriptors
        if pattern_descriptors is None:
            self.pattern_descriptors = {}
        else:
            check_descriptors_dimension(pattern_descriptors,
                                        'pattern_descriptors',
                                        self.n_cond)
            self.pattern_descriptors = pattern_descriptors
        self.dissimilarity_measure = dissimilarity_measure

    def __repr__(self):
        """
        defines string which is printed for the object
        """
        return (f'pyrsa.rdm.{self.__class__.__name__}(\n'
                f'dissimilarity_measure = \n{self.dissimilarity_measure}\n'
                f'dissimilarities = \n{self.dissimilarities}\n'
                f'descriptors = \n{self.descriptors}\n'
                f'rdm_descriptors = \n{self.rdm_descriptors}\n'
                f'pattern_descriptors = \n{self.pattern_descriptors}\n'
                )

    def __str__(self):
        """
        defines the output of print
        """
        string_desc = format_descriptor(self.descriptors)
        rdm_desc = format_descriptor(self.rdm_descriptors)
        pattern_desc = format_descriptor(self.pattern_descriptors)
        diss = self.get_matrices()[0]
        return (f'pyrsa.rdm.{self.__class__.__name__}\n'
                f'{self.n_rdm} RDM(s) over {self.n_cond} conditions\n\n'
                f'dissimilarity_measure = \n{self.dissimilarity_measure}\n\n'
                f'dissimilarities[0] = \n{diss}\n\n'
                f'descriptors: \n{string_desc}\n'
                f'rdm_descriptors: \n{rdm_desc}\n'
                f'pattern_descriptors: \n{pattern_desc}\n'
                )

    def get_vectors(self):
        """ Returns RDMs as np.ndarray with each RDM as a vector

        Returns:
            numpy.ndarray: RDMs as a matrix with one row per RDM

        """
        return self.dissimilarities

    def get_matrices(self):
        """ Returns RDMs as np.ndarray with each RDM as a matrix

        Returns:
            numpy.ndarray: RDMs as a 3-Tensor with one matrix per RDM

        """
        matrices, _, _ = batch_to_matrices(self.dissimilarities)
        return matrices

    def subset_pattern(self, by, value):
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
        dissimilarities = self.get_matrices()[:, selection][:, :, selection]
        descriptors = self.descriptors
        pattern_descriptors = extract_dict(
            self.pattern_descriptors, selection)
        rdm_descriptors = self.rdm_descriptors
        rdms = RDMs(dissimilarities=dissimilarities,
                    descriptors=descriptors,
                    rdm_descriptors=rdm_descriptors,
                    pattern_descriptors=pattern_descriptors)
        return rdms

    def subsample_pattern(self, by, value):
        """ Returns a subsampled RDMs with repetitions if values are repeated
        
        Args:
            by(String): the descriptor by which the subset selection
                        is made from descriptors
            value:      the value by which the subset selection is made
                        from descriptors

        Returns:
            RDMs object, with subsampled patterns

        """   
        if (type(value) is list or
            type(value) is tuple or
            type(value) is np.ndarray):
            selection = [np.asarray(self.pattern_descriptors[by]==i).nonzero()[0]
                         for i in value]
            selection = np.concatenate(selection)
        else:
            selection = np.where(self.rdm_descriptors[by]==value)
        dissimilarities = self.get_matrices()[:, selection][:, :, selection]
        descriptors = self.descriptors
        pattern_descriptors = extract_dict(
            self.pattern_descriptors, selection)
        rdm_descriptors = self.rdm_descriptors
        rdms = RDMs(dissimilarities=dissimilarities,
                    descriptors=descriptors,
                    rdm_descriptors=rdm_descriptors,
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
        selection = bool_index(self.rdm_descriptors[by], value)
        dissimilarities = self.dissimilarities[selection, :]
        descriptors = self.descriptors
        pattern_descriptors = self.pattern_descriptors
        rdm_descriptors = extract_dict(self.rdm_descriptors, selection)
        rdms = RDMs(dissimilarities=dissimilarities,
                    descriptors=descriptors,
                    rdm_descriptors=rdm_descriptors,
                    pattern_descriptors=pattern_descriptors)
        return rdms

    def subsample(self, by, value):
        """ Returns a subsampled RDMs with repetitions if values are repeated
        
        Args:
            by(String): the descriptor by which the subset selection
                        is made from descriptors
            value:      the value by which the subset selection is made
                        from descriptors

        Returns:
            RDMs object, with subsampled RDMs

        """   
        if (type(value) is list or
            type(value) is tuple or
            type(value) is np.ndarray):
            selection = [np.asarray(self.rdm_descriptors[by]==i).nonzero()[0]
                         for i in value]
            selection = np.concatenate(selection)
        else:
            selection = np.where(self.rdm_descriptors[by]==value)
        dissimilarities = self.dissimilarities[selection, :]
        descriptors = self.descriptors
        pattern_descriptors = self.pattern_descriptors
        rdm_descriptors = extract_dict(self.rdm_descriptors, selection)
        rdms = RDMs(dissimilarities=dissimilarities,
                    descriptors=descriptors,
                    rdm_descriptors=rdm_descriptors,
                    pattern_descriptors=pattern_descriptors)
        return rdms
