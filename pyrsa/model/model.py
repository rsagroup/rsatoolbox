#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Definition of RSA Model class and subclasses
@author: jdiedrichsen, heiko
"""

import numpy as np
from pyrsa.rdm import RDMs
from pyrsa.util.rdm_utils import batch_to_vectors


class Model:
    """
    Abstract model class.
    Defines members that every class needs to have, but does not implement any
    interesting behavior. Inherit from this class to define specific model
    types
    """
    def __init__(self, name):
        self.name = name
        self.n_param = 0
        self.default_fitter = None

    def predict(self, theta=None):
        """ Returns the predicted rdm vector

        Args:
            theta(numpy.ndarray): the model parameter vector (one dimensional)

        Returns:
            numpy.ndarray: rdm vector
        """
        raise NotImplementedError(
            "Predict function not implemented in used model class!"
            )

    def predict_rdm(self, theta=None):
        """ Returns the predicted rdm as an object

        Args:
            theta(numpy.ndarray): the model parameter vector (one dimensional)

        Returns:
            numpy.ndarray: rdm object
        """
        raise NotImplementedError(
            "Predict rdm function not implemented in used model class!"
            )

    def fit(self, data):
        """ fit the model to a RDM object data

        Args:
            data(RDM object): the RDMs to be fit with the model

        Returns:
            theta(numpy.ndarray): parameter vector (one dimensional)
        """
        return np.array([])


class ModelFixed(Model):
    def __init__(self, name, rdm):
        """
        Fixed model
        This is a parameter-free model that simply predicts a fixed RDM
        It takes rdm object, a vector or a matrix as input to define the RDM
        
        Args:
            Name(String): Model name
            rdm(pyrsa.rdm.RDMs): rdms in one object
        """
        Model.__init__(self, name)
        if isinstance(rdm, RDMs):
            self.rdm_obj = rdm
            self.rdm = np.mean(rdm.get_vectors(), axis=0)
        elif rdm.ndim == 1:  # User passed a vector
            self.rdm_obj = RDMs(np.array([rdm]))
            self.n_cond = (1 + np.sqrt(1 + 8 * rdm.size)) / 2
            if self.n_cond % 1 != 0:
                raise NameError(
                    "RDM vector needs to have size of ncond*(ncond-1)/2"
                    )
            self.rdm = rdm
        else: # User passed a matrix
            self.rdm_obj = RDMs(np.array([rdm]))
            self.rdm = batch_to_vectors(np.array([rdm]))[0]
        self.n_param = 0

    def predict(self, theta=None):
        """ Returns the predicted rdm vector

        For the fixed model there are no parameters. theta is ignored.

        Args:
            theta(numpy.ndarray): the model parameter vector (one dimensional)

        Returns:
            rdm vector

        """
        return self.rdm

    def predict_rdm(self, theta=None):
        """ Returns the predicted rdm vector

        For the fixed model there are no parameters.

        Args:
            theta(numpy.ndarray): the model parameter vector (one dimensional)

        Returns:
            pyrsa.rdm.RDMs: rdm object

        """
        return self.rdm_obj


class ModelSelect(Model):
    """
    Selection model
    This model has a set of RDMs and selects one of them as its prediction.
    theta should here be an integer index
    """
    # Model Constructor
    def __init__(self, name, rdm):
        Model.__init__(self, name)
        if isinstance(rdm, RDMs):
            self.rdm_obj = rdm
            self.rdm = rdm.get_vectors()
        elif rdm.ndim == 2:  # User supplied vectors
            self.rdm_obj = RDMs(np.array([rdm]))
            self.n_cond = (1 + np.sqrt(1 + 8 * rdm.size)) / 2
            if self.n_cond % 1 != 0:
                raise NameError(
                    "RDM vector needs to have size of ncond*(ncond-1)/2"
                    )
            self.rdm = rdm
        else: # User passed matrixes
            self.rdm_obj = RDMs(rdm)
            self.rdm = batch_to_vectors(rdm)
        self.n_param = 1
        self.n_rdm = self.rdm_obj.n_rdm

    def predict(self, theta=0):
        """ Returns the predicted rdm vector

        For the fixed model there are no parameters. theta is ignored.

        Args:
            theta(numpy.ndarray): the model parameter vector (one dimensional)

        Returns:
            rdm vector

        """
        return self.rdm[theta]

    def predict_rdm(self, theta=0):
        """ Returns the predicted rdm vector

        For the fixed model there are no parameters.

        Args:
            theta(numpy.ndarray): the model parameter vector (one dimensional)

        Returns:
            pyrsa.rdm.RDMs: rdm object

        """
        return self.rdm_obj[theta]
