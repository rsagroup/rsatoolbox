#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Definition of RSA RDMs class and subclasses

@author: baihan
"""
from __future__ import annotations
from typing import Dict, Optional
import warnings
from copy import deepcopy
from collections.abc import Iterable
import numpy as np
from rsatoolbox.io.pandas import rdms_to_df
from rsatoolbox.rdm.combine import _mean
from rsatoolbox.util.rdm_utils import batch_to_vectors
from rsatoolbox.util.rdm_utils import batch_to_matrices
from rsatoolbox.util.descriptor_utils import format_descriptor
from rsatoolbox.util.descriptor_utils import num_index
from rsatoolbox.util.descriptor_utils import subset_descriptor
from rsatoolbox.util.descriptor_utils import check_descriptor_length_error
from rsatoolbox.util.descriptor_utils import append_descriptor
from rsatoolbox.util.descriptor_utils import dict_to_list
from rsatoolbox.util.descriptor_utils import desc_eq
from rsatoolbox.util.data_utils import extract_dict
from rsatoolbox.rdm.combine import _merged_rdm_descriptors
from rsatoolbox.io.hdf5 import read_dict_hdf5, write_dict_hdf5
from rsatoolbox.io.pkl import read_dict_pkl, write_dict_pkl
from rsatoolbox.util.file_io import remove_file


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
    dissimilarities: np.ndarray
    dissimilarity_measure: Optional[str]
    descriptors: Dict
    rdm_descriptors: Dict
    pattern_descriptors: Dict

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
            for k, v in rdm_descriptors.items():
                if not isinstance(v, Iterable) or isinstance(v, str):
                    rdm_descriptors[k] = [v]
            check_descriptor_length_error(rdm_descriptors,
                                          'rdm_descriptors',
                                          self.n_rdm)
            self.rdm_descriptors = rdm_descriptors
        if pattern_descriptors is None:
            self.pattern_descriptors = {}
        else:
            for k, v in pattern_descriptors.items():
                if not isinstance(v, Iterable) or isinstance(v, str):
                    pattern_descriptors[k] = [v]
            check_descriptor_length_error(pattern_descriptors,
                                          'pattern_descriptors',
                                          self.n_cond)
            self.pattern_descriptors = pattern_descriptors
        if 'index' not in self.pattern_descriptors.keys():
            self.pattern_descriptors['index'] = list(range(self.n_cond))
        if 'index' not in self.rdm_descriptors.keys():
            self.rdm_descriptors['index'] = list(range(self.n_rdm))
        self.dissimilarity_measure = dissimilarity_measure

    def __repr__(self):
        """
        defines string which is printed for the object
        """
        return (f'rsatoolbox.rdm.{self.__class__.__name__}(\n'
                f'dissimilarity_measure = \n{self.dissimilarity_measure}\n'
                f'dissimilarities = \n{self.dissimilarities}\n'
                f'descriptors = \n{self.descriptors}\n'
                f'rdm_descriptors = \n{self.rdm_descriptors}\n'
                f'pattern_descriptors = \n{self.pattern_descriptors}\n'
                )

    def __eq__(self, other: object) -> bool:
        """Test for equality
        This magic method gets called when you compare two
        RDMs objects: `rdms1 == rdms2`.
        True if the objects are of the same type, and
        dissimilarities and descriptors are equal.

        Args:
            other (RDMs): The second RDMs object to
                compare this one with

        Returns:
            bool: True if equal
        """
        if isinstance(other, RDMs):
            return all([
                np.all(self.dissimilarities == other.dissimilarities),
                self.descriptors == other.descriptors,
                desc_eq(self.rdm_descriptors, other.rdm_descriptors),
                desc_eq(self.pattern_descriptors, other.pattern_descriptors),
            ])
        return False

    def __str__(self):
        """
        defines the output of print
        """
        string_desc = format_descriptor(self.descriptors)
        rdm_desc = format_descriptor(self.rdm_descriptors)
        pattern_desc = format_descriptor(self.pattern_descriptors)
        diss = self.get_matrices()[0]
        return (f'rsatoolbox.rdm.{self.__class__.__name__}\n'
                f'{self.n_rdm} RDM(s) over {self.n_cond} conditions\n\n'
                f'dissimilarity_measure = \n{self.dissimilarity_measure}\n\n'
                f'dissimilarities[0] = \n{diss}\n\n'
                f'descriptors: \n{string_desc}\n'
                f'rdm_descriptors: \n{rdm_desc}\n'
                f'pattern_descriptors: \n{pattern_desc}\n'
                )

    def __getitem__(self, idx):
        """
        allows indexing with []
        and iterating over RDMs with `for rdm in rdms:`
        """
        dissimilarities = self.dissimilarities[np.array(idx)].reshape(
            -1, self.dissimilarities.shape[1])
        rdm_descriptors = subset_descriptor(self.rdm_descriptors, idx)
        rdms = RDMs(dissimilarities,
                    dissimilarity_measure=self.dissimilarity_measure,
                    descriptors=self.descriptors,
                    rdm_descriptors=rdm_descriptors,
                    pattern_descriptors=self.pattern_descriptors)
        return rdms

    def __len__(self) -> int:
        """
        The number of RDMs in this stack.
        Together with __getitem__, allows `reversed(rdms)`.
        """
        return self.n_rdm

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

    def copy(self) -> RDMs:
        """Return a copy of this object, with all properties
        equal to the original's

        Returns:
            RDMs: Value copy
        """
        return RDMs(
            dissimilarities=self.dissimilarities.copy(),
            dissimilarity_measure=self.dissimilarity_measure,
            descriptors=deepcopy(self.descriptors),
            rdm_descriptors=deepcopy(self.rdm_descriptors),
            pattern_descriptors=deepcopy(self.pattern_descriptors)
        )

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
        if by is None:
            by = 'index'
        if not isinstance(value, Iterable):
            value = [value]
        selection = num_index(self.pattern_descriptors[by], value)
        ix, iy = np.triu_indices(self.n_cond, 1)
        pattern_in_value = np.array(
            [p in value for p in self.pattern_descriptors[by]])
        selection_xy = pattern_in_value[ix] & pattern_in_value[iy]
        dissimilarities = self.dissimilarities[:, selection_xy]
        descriptors = self.descriptors
        pattern_descriptors = extract_dict(
            self.pattern_descriptors, selection)
        rdm_descriptors = self.rdm_descriptors
        dissimilarity_measure = self.dissimilarity_measure
        rdms = RDMs(dissimilarities=dissimilarities,
                    descriptors=descriptors,
                    rdm_descriptors=rdm_descriptors,
                    pattern_descriptors=pattern_descriptors,
                    dissimilarity_measure=dissimilarity_measure)
        return rdms

    def subsample_pattern(self, by, value):
        """ Returns a subsampled RDMs with repetitions if values are repeated

        This function now generates Nans where the off-diagonal 0s would
        appear. These values are trivial to predict for models and thus
        need to be marked and excluded from the evaluation.

        Args:
            by(String): the descriptor by which the subset selection
                        is made from descriptors
            value:      the value(s) by which the subset selection is made
                        from descriptors

        Returns:
            RDMs object, with subsampled patterns

        """
        if by is None:
            by = 'index'
        desc = np.array(self.pattern_descriptors[by])  # desc is list-like
        if isinstance(value, (list, tuple, np.ndarray)):
            selection = [np.asarray(desc == i).nonzero()[0]
                         for i in value]
            selection = np.concatenate(selection)
        else:
            selection = np.where(desc == value)[0]
        selection = np.sort(selection)
        dissimilarities = self.get_matrices()
        for i_rdm in range(self.n_rdm):
            np.fill_diagonal(dissimilarities[i_rdm], np.nan)
        selection = np.sort(selection)
        dissimilarities = dissimilarities[:, selection][:, :, selection]
        descriptors = self.descriptors
        pattern_descriptors = extract_dict(
            self.pattern_descriptors, selection)
        rdm_descriptors = self.rdm_descriptors
        dissimilarity_measure = self.dissimilarity_measure
        rdms = RDMs(dissimilarities=dissimilarities,
                    descriptors=descriptors,
                    rdm_descriptors=rdm_descriptors,
                    pattern_descriptors=pattern_descriptors,
                    dissimilarity_measure=dissimilarity_measure)
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
        if by is None:
            by = 'index'
        selection = num_index(self.rdm_descriptors[by], value)
        dissimilarities = self.dissimilarities[selection, :]
        descriptors = self.descriptors
        pattern_descriptors = self.pattern_descriptors
        rdm_descriptors = extract_dict(self.rdm_descriptors, selection)
        dissimilarity_measure = self.dissimilarity_measure
        rdms = RDMs(dissimilarities=dissimilarities,
                    descriptors=descriptors,
                    rdm_descriptors=rdm_descriptors,
                    pattern_descriptors=pattern_descriptors,
                    dissimilarity_measure=dissimilarity_measure)
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
        if by is None:
            by = 'index'
        desc = self.rdm_descriptors[by]
        selection = []
        if isinstance(value, (list, tuple, np.ndarray)):
            for i in value:
                for j, d in enumerate(desc):
                    if d == i:
                        selection.append(j)
        else:
            for j, d in enumerate(desc):
                if d == value:
                    selection.append(j)
        dissimilarities = self.dissimilarities[selection, :]
        descriptors = self.descriptors
        pattern_descriptors = self.pattern_descriptors
        rdm_descriptors = extract_dict(self.rdm_descriptors, selection)
        dissimilarity_measure = self.dissimilarity_measure
        rdms = RDMs(dissimilarities=dissimilarities,
                    descriptors=descriptors,
                    rdm_descriptors=rdm_descriptors,
                    pattern_descriptors=pattern_descriptors,
                    dissimilarity_measure=dissimilarity_measure)
        return rdms

    def append(self, rdm):
        """ appends an rdm to the object
        The rdm should have the same shape and type as this object.
        Its pattern_descriptor and descriptor are ignored

        Args:
            rdm(rsatoolbox.rdm.RDMs): the rdm to append

        Returns:

        """
        assert isinstance(rdm, RDMs), 'appended rdm should be an RDMs'
        assert rdm.n_cond == self.n_cond, 'appended rdm had wrong shape'
        assert rdm.dissimilarity_measure == self.dissimilarity_measure, \
            'appended rdm had wrong dissimilarity measure'
        self.dissimilarities = np.concatenate((
            self.dissimilarities, rdm.dissimilarities), axis=0)
        self.rdm_descriptors = append_descriptor(self.rdm_descriptors,
                                                 rdm.rdm_descriptors)
        self.n_rdm = self.n_rdm + rdm.n_rdm

    def save(self, filename, file_type='hdf5', overwrite=False):
        """ saves the RDMs object into a file

        Args:
            filename(String): path to file to save to
                [or opened file]
            file_type(String): Type of file to create:
                hdf5: hdf5 file
                pkl: pickle file
            overwrite(Boolean): overwrites file if it already exists

        """
        rdm_dict = self.to_dict()
        if overwrite:
            remove_file(filename)
        if file_type == 'hdf5':
            write_dict_hdf5(filename, rdm_dict)
        elif file_type == 'pkl':
            write_dict_pkl(filename, rdm_dict)

    def to_dict(self):
        """ converts the object into a dictionary, which can be saved to disk

        Returns:
            rdm_dict(dict): dictionary containing all information required to
                recreate the RDMs object
        """
        rdm_dict = {}
        rdm_dict['dissimilarities'] = self.dissimilarities
        rdm_dict['descriptors'] = self.descriptors
        rdm_dict['rdm_descriptors'] = self.rdm_descriptors
        rdm_dict['pattern_descriptors'] = self.pattern_descriptors
        rdm_dict['dissimilarity_measure'] = self.dissimilarity_measure
        return rdm_dict

    def to_df(self):
        """Return a new long-form pandas DataFrame representing this RDM

        See `rsatoolbox.io.pandas.rdms_to_df` for details

        Returns:
            pandas.DataFrame: The DataFrame for this RDMs object
        """
        return rdms_to_df(self)

    def reorder(self, new_order):
        """Reorder the patterns according to the index in new_order

        Args:
            new_order (numpy.ndarray): new order of patterns,
                vector of length equal to the number of patterns
        """
        matrices = self.get_matrices()
        matrices = matrices[(slice(None),) + np.ix_(new_order, new_order)]
        self.dissimilarities = batch_to_vectors(matrices)[0]
        for dname, descriptors in self.pattern_descriptors.items():
            self.pattern_descriptors[dname] = [descriptors[idx] for idx in new_order]

    def sort_by(self, reindex: bool = True, **kwargs):
        """Reorder the patterns by sorting a descriptor

        Args:
            reindex (bool): whether to reset the 'index' descriptor
                following sorting

        Pass keyword arguments that correspond to descriptors,
        with value indicating the sort type. Supported methods:

            'alpha': sort alphabetically (using np.sort)

            list/np.array: specify the new order explicitly. Values should
                correspond to the descriptor values

        Examples:

            The following code sorts the 'condition' descriptor alphabetically:

            ::

                rdms.sort_by(condition='alpha')

            The following code sort the 'condition' descriptor in the order
            1, 3, 2, 4, 5:

            ::

                rdms.sort_by(condition=[1, 3, 2, 4, 5])

        Raises:
            ValueError: Raised if the method chosen is not implemented
        """
        for dname, method in kwargs.items():
            if method == 'alpha':
                descriptor = self.pattern_descriptors[dname]
                self.reorder(np.argsort(descriptor, kind='stable'))
            elif isinstance(method, (list, np.ndarray)):
                # in this case, `method` is the desired descriptor order
                new_order = method
                descriptor = self.pattern_descriptors[dname]
                if not set(descriptor).issubset(new_order):
                    raise ValueError(f'Expected {method} to be a permutation \
                            or subset of {descriptor}')
                # convert to indices to use `reorder` method
                self.reorder([list(descriptor).index(x) for x in new_order])
            else:
                raise ValueError(f'Unknown sorting method: {method}')
        if reindex:
            self.pattern_descriptors['index'] = list(range(self.n_cond))

    def mean(self, weights=None):
        """Average rdm of all rdms contained

        Args:
            weights (str or ndarray, optional): One of:
                None: No weighting applied
                str: Use the weights contained in the `rdm_descriptor` with this name
                ndarray: Weights array of the shape of RDMs.dissimilarities

        Returns:
            `rsatoolbox.rdm.rdms.RDMs`: New RDMs object with one vector
        """
        if str(weights) in self.rdm_descriptors:
            new_descriptors = {
                (k, v) for (k, v) in self.descriptors.items() if k != weights
            }
            weights = self.rdm_descriptors[weights]
        else:
            new_descriptors = deepcopy(self.descriptors)
        return RDMs(
            dissimilarities=np.array([_mean(self.dissimilarities, weights)]),
            dissimilarity_measure=self.dissimilarity_measure,
            descriptors=new_descriptors,
            pattern_descriptors=deepcopy(self.pattern_descriptors)
        )


def rdms_from_dict(rdm_dict):
    """ creates a RDMs object from a dictionary

    Args:
        rdm_dict(dict): dictionary with information

    Returns:
        rdms(RDMs): the regenerated RDMs object

    """
    rdms = RDMs(dissimilarities=rdm_dict['dissimilarities'],
                descriptors=rdm_dict['descriptors'],
                rdm_descriptors=dict_to_list(rdm_dict['rdm_descriptors']),
                pattern_descriptors=dict_to_list(rdm_dict['pattern_descriptors']),
                dissimilarity_measure=rdm_dict['dissimilarity_measure'])
    return rdms


def load_rdm(filename, file_type=None):
    """ loads a RDMs object from disk

    Args:
        filename(String): path to file to load

    """
    if file_type is None:
        if isinstance(filename, str):
            if filename[-4:] == '.pkl':
                file_type = 'pkl'
            elif filename[-3:] == '.h5' or filename[-4:] == 'hdf5':
                file_type = 'hdf5'
    if file_type == 'hdf5':
        rdm_dict = read_dict_hdf5(filename)
    elif file_type == 'pkl':
        rdm_dict = read_dict_pkl(filename)
    else:
        raise ValueError('filetype not understood')
    return rdms_from_dict(rdm_dict)


def concat(*rdms: RDMs, target_pdesc: Optional[str] = None) -> RDMs:
    """Merge into single RDMs object
    requires that the rdms have the same shape
    descriptor and pattern descriptors are taken from the first rdms object
    for rdm_descriptors concatenation is tried
    the rdm index is reinitialized

    Args:
        rdms(iterable of rsatoolbox.rdm.RDMs): RDMs objects to be concatenated
            or multiple RDMs as separate arguments
        target_pdesc(optional, str): a pattern descriptor to use for sorting

    Returns:
        rsatoolbox.rdm.RDMs: concatenated rdms object

    """
    if len(rdms) == 1: ## single argument
        if isinstance(rdms[0], RDMs):
            rdms_list = [rdms[0]]
        else:
            rdms_list = list(rdms[0])
    else: ## multiple arguments
        rdms_list = list(rdms)
    assert isinstance(rdms_list[0], RDMs), \
        'Supply list of RDMs objects, or RDMs objects as separate arguments'

    descriptors, rdm_descriptors = _merged_rdm_descriptors(rdms_list)

    if target_pdesc is None:
        # see if we can find an authoritative descriptor for pattern order
        pdescs = rdms_list[0].pattern_descriptors.keys()
        pdesc_candidates = list(filter(
            lambda n: n != 'index' and (
                len(rdms_list[0].pattern_descriptors[n])
                == len(set(rdms_list[0].pattern_descriptors[n]))),
                pdescs))
        target_pdesc = None
        if len(pdesc_candidates) > 0:
            target_pdesc = pdesc_candidates[0]
        if len(pdesc_candidates) > 1:
            warnings.warn(f'[concat] Multiple pattern descriptors found, using "{target_pdesc}"')
    else:
        assert target_pdesc in rdms_list[0].pattern_descriptors.keys(), \
            'The provided descriptor is not a pattern descriptor'
        assert len(rdms_list[0].pattern_descriptors[target_pdesc]) == rdms_list[0].n_cond, \
            'The provided descriptor is not unique'

    for rdm_new in rdms_list[1:]:
        assert isinstance(rdm_new, RDMs), 'rdm for concat should be an RDMs'
        assert rdm_new.n_cond == rdms_list[0].n_cond, 'rdm for concat had wrong shape'
        assert rdm_new.dissimilarity_measure == rdms_list[0].dissimilarity_measure, \
            'appended rdm had wrong dissimilarity measure'
        if target_pdesc:
            # if we have a target descriptor, check if the order is the same
            auth_order = rdms_list[0].pattern_descriptors[target_pdesc]
            other_order = rdm_new.pattern_descriptors[target_pdesc]
            if not np.all(other_order == auth_order):
                # order varies; reorder this rdms object
                _, new_order = np.where(auth_order[:, None] == other_order)
                rdm_new.reorder(new_order)

    dissimilarities = np.concatenate([
        rdm.dissimilarities
        for rdm in rdms_list
        ], axis=0)
    # Set dissimilarity measure if it's the same for all rdms in list
    if len(set(r.dissimilarity_measure for r in rdms_list)) == 1:
        dissimilarity_measure = rdms_list[0].dissimilarity_measure
    else:
        dissimilarity_measure = None
    rdm = RDMs(
        dissimilarities=dissimilarities,
        dissimilarity_measure=dissimilarity_measure,
        rdm_descriptors=rdm_descriptors,
        descriptors=descriptors,
        pattern_descriptors=rdms_list[0].pattern_descriptors
    )
    return rdm


def permute_rdms(rdms, p=None):
    """ Permute rows, columns and corresponding pattern descriptors
    of RDM matrices according to a permutation vector

    Args:
        p (numpy.ndarray):
           permutation vector (values must be unique integers
           from 0 to n_cond of RDM matrix).
           If p = None, a random permutation vector is created.

    Returns:
        rdm_p(rsatoolbox.rdm.RDMs): the rdm object with a permuted matrix
            and pattern descriptors

    """
    if p is None:
        p = np.random.permutation(rdms.n_cond)
        print('No permutation vector specified,'
              + ' performing random permutation.')

    assert p.dtype == 'int', "permutation vector must have integer entries."
    assert min(p) == 0 and max(p) == rdms.n_cond-1, \
        "permutation vector must have entries ranging from 0 to n_cond"
    assert len(np.unique(p)) == rdms.n_cond, \
        "permutation vector must only have unique integer entries"

    rdm_mats = rdms.get_matrices()
    descriptors = rdms.descriptors.copy()
    rdm_descriptors = rdms.rdm_descriptors.copy()
    pattern_descriptors = rdms.pattern_descriptors.copy()

    # To easily reverse permutation later
    p_inv = np.arange(len(p))[np.argsort(p)]
    descriptors.update({'p_inv': p_inv})
    rdm_mats = rdm_mats[:, p, :]
    rdm_mats = rdm_mats[:, :, p]
    stims = np.array(pattern_descriptors['index'])
    pattern_descriptors.update({'index': list(stims[p].astype(np.str_))})

    rdms_p = RDMs(
        dissimilarities=rdm_mats,
        descriptors=descriptors,
        rdm_descriptors=rdm_descriptors,
        pattern_descriptors=pattern_descriptors)
    return rdms_p


def inverse_permute_rdms(rdms):
    """ Gimmick function to reverse the effect of permute_rdms() """

    p_inv = rdms.descriptors['p_inv']
    rdms_p = permute_rdms(rdms, p=p_inv)
    return rdms_p


def get_categorical_rdm(category_vector, category_name='category'):
    """ generates an RDM object containing a categorical RDM, i.e. RDM = 0
    if the category is the same and 1 if they are different

    Args:
        category_vector(iterable): a category index per condition
        category_name(String): name for the descriptor in the object, defaults
            to 'category'

    Returns:
        rsatoolbox.rdm.RDMs: constructed RDM

    """
    n = len(category_vector)
    rdm_list = []
    for i_cat in range(n):
        for j_cat in range(i_cat + 1, n):
            if isinstance(category_vector[i_cat], Iterable):
                comparisons = [np.array(category_vector[i_cat][idx])
                               != np.array(category_vector[j_cat][idx])
                               for idx in range(len(category_vector[i_cat]))]
                rdm_list.append(np.any(comparisons))
            else:
                rdm_list.append(
                    category_vector[i_cat] != category_vector[j_cat])
    rdm = RDMs(np.array(rdm_list, dtype=float),
               pattern_descriptors={category_name: np.array(category_vector)})
    return rdm
