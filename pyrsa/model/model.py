#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Definition of RSA Model class and subclasses
@author: jdiedrichsen, heiko
"""

import numpy as np



class Model:
    """
    Abstract model class.
    Defines members that every class needs to have, but does not implement any
    interesting behavior. Inherit from this class to define specific model
    types
    """
    # Model Constructor
    def __init__(self, name):
        self.name = name
        self.n_param = 0

    # The predict function should return a rdm vector
    def predict(self, theta):
        """ Returns the predicted rdm(-vector)

        Args:
            theta(np.array 1d): the model parameter vector

        Returns:
            rdm vector
        """
        raise NotImplementedError("Predict function not implemented in used model class!")
    def fit(self, data):
        """ fit the model to a RDM object data

        Args:
            data(RDM object): the RDMs to be fit with the model

        Returns:
            theta(np.array 1d): parameter vector
        """


class ModelFixed(Model):
    """
    Fixed model
    This is a parameter-free model that simply predicts a fixed RDM
    It takes a unidimension numpy-vector as an input to define the RDM
    """
    # Model Constructor
    def __init__(self, name, rdm):
        Model.__init__(self, name)
        if rdm.ndim == 1:  # User passed a vector
            self.n_cond = (1+np.sqrt(1+8*rdm.size))/2
            if self.n_cond%1 != 0:
                raise NameError("RDM vector needs to have size of ncond*(ncond-1)/2")
            self.rdm = rdm   # Add check to make sure it's
            self.n_param = 0

    # prediction returns the predicted rdm vector
    def predict(self, theta=None):
        """ Returns the predicted rdm(-vector)

        For the fixed model there are no parameters.

        Args:
            theta(np.array 1d): the model parameter vector

        Returns:
            rdm vector
        """
        return self.rdm
