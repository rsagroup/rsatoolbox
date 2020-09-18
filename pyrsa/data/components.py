#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Computes and projects data onto a subspace defined by a set of components.
RDMs can then be visualized in that subspace.

See demo_component_projection.ipynb

@author: snormanhaignere
"""

import numpy as np
import sklearn


class Components:
    '''Component model of data matrix.

    Data matrix is approximated as the product of a response
    and weight matrix (D ~= R * W)

    Contains functions for computing component decompositions
    and for reconstructing data given those component decompositions

    Attributes:
        R: [stimuli x component]
        W: [component x channel]
    '''

    def __init__(self, R=None, W=None):
        '''Constructs the core component object.

        When called with no arguments, R and W are simply set to None.
        Once initialized, you can computed components from data
        using one of the component functions:
            pca
            fastica

        Args:
            R: Optional; [stimuli x component] numpy array
            W: Optional; [component x channel] numpy array
        '''

        self.R = R
        self.W = W
        if self.R is not None:
            if self.R.shape[1] != self.W.shape[0]:
                raise NameError('Columns of R must match rows of W')
            self.n_components = R.shape[1]
        else:
            self.n_components = None

    def reconstruct(self, subset=None):
        '''Reconstructs data by multiplying response and weight matrices.

        Args:
            subset: Optional; list of component indices.
              Default is to use all components

        Returns:
            A [stimuli x channel] reconstruction of your data.
        '''

        if subset is not None:
            measurements = np.matmul(self.R[:, subset], self.W[subset, :])
        else:
            measurements = np.matmul(self.R, self.W)
        return measurements

    def pca(self, measurements, n_components=None):
        '''PCA decomposition of data matrix.

        Decomposition is computed using SVD:
            D = USV

        Components are then given by:
            R = U
            W = SV

        Args:
            measurements: [stimuli x channel] numpy array
            n_components: Optional; if specified, only returns top N PCs:
              R = R[:, :n_components]
              W = W[:n_components, :]

        Returns:
            Objects with self.R and self.W given as above
        '''
        [U, s, Vh] = np.linalg.svd(measurements, full_matrices=False)
        self.R = U
        self.W = np.expand_dims(s, axis=1) * Vh
        self._select_top_components(n_components)

    def fastica(self, measurements, n_components=None, method_params=None):
        '''Decomposition computed using FastICA.

        Returns decomposition with maximally independent weights,
        as estimated using FastICA. Unlike PCA, component responses
        can be correlated. In the language of ICA, R is the 'mixing matrix'.

        Note that unlike standard ICA, the weights returned are not
        zero-mean. The weights returned are given by the product of the
        response profile matrix (mixing matrix) with the data matrix,
        which provides a way to infer a meaningful mean.

        Data can be first projected onto a low-dimensional subspace
        using PCA, which is often a good idea (see n_components argument)

        Decomposition performed using sklearn:
            sklearn.decomposition.FastICA

        Args:
            measurements: [stimuli x channel] numpy array
            n_components: Optional; if specified, data is first
              reduced in dimensionality using PCA, and then rotated
              to maximize independence within this subspace.

        Returns:
            Objects with self.R and self.W given as above
        '''

        if method_params is None:
            method_params = {}
        ica = sklearn.decomposition.FastICA(
            n_components=n_components, **method_params)
        self.R = ica.mixing_
        self.W = np.matmul(np.linalg.pinv(self.R), measurements)

    def _select_top_components(self, n_components=None):
        if n_components is not None:
            self.R = self.R[:, :n_components]
            self.W = self.W[:n_components, :]
            self.n_components = n_components

    def order_components(self, order):
        '''Re-order components'''
        self.R = self.R[:, order]
        self.W = self.W[order, :]
