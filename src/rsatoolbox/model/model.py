#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Definition of RSA Model class and subclasses
"""

import numpy as np
from rsatoolbox.rdm import RDMs
from rsatoolbox.rdm import rdms_from_dict
from rsatoolbox.util.rdm_utils import batch_to_vectors
from .fitter import fit_mock, fit_optimize, fit_select, fit_interpolate


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
        self.default_fitter = fit_mock
        self.rdm_obj = None

    def predict(self, theta=None):
        """ Returns the predicted rdm vector

        Args:
            theta(numpy.ndarray): the model parameter vector (one dimensional)

        Returns:
            numpy.ndarray: rdm vector
        """
        raise NotImplementedError(
            "Predict function not implemented in used model class!")

    def predict_rdm(self, theta=None):
        """ Returns the predicted rdm as an object

        Args:
            theta(numpy.ndarray): the model parameter vector (one dimensional)

        Returns:
            numpy.ndarray: rdm object
        """
        raise NotImplementedError(
            "Predict rdm function not implemented in used model class!")

    def fit(self, data, method='cosine', pattern_idx=None,
            pattern_descriptor=None, sigma_k=None):
        """ fit the model to a RDM object data

        Args:
            data(RDM object): the RDMs to be fit with the model
            method(String): how to measure rdm_similarity
            patterrn_idx: which patterns to use
            pattern_descriptor: which part of the dict to use to interpret
                pattern_idx

        Returns:
            theta(numpy.ndarray): parameter vector (one dimensional)
        """
        return self.default_fitter(self, data, method=method,
                                   pattern_idx=pattern_idx,
                                   pattern_descriptor=pattern_descriptor,
                                   sigma_k=sigma_k)

    def to_dict(self):
        """ Converts the model into a dictionary, which can be used for saving

        Returns:
            model_dict(dict): A dictionary containting all data needed to
                recreate the object

        """
        model_dict = {}
        if self.rdm_obj:
            model_dict['rdm'] = self.rdm_obj.to_dict()
        else:
            model_dict['rdm'] = None
        model_dict['name'] = self.name
        model_dict['type'] = type(self).__name__
        return model_dict


class ModelFixed(Model):
    def __init__(self, name, rdm):
        """
        Fixed model
        This is a parameter-free model that simply predicts a fixed RDM
        It takes rdm object, a vector or a matrix as input to define the RDM

        Args:
            Name(String): Model name
            rdm(rsatoolbox.rdm.RDMs): rdms in one object
        """
        Model.__init__(self, name)
        if isinstance(rdm, RDMs):
            self.rdm_obj = rdm
            self.rdm = np.mean(rdm.get_vectors(), axis=0)
            self.n_cond = rdm.n_cond
        elif rdm.ndim == 1:  # User passed a vector
            self.rdm_obj = RDMs(np.array([rdm]))
            self.n_cond = (1 + np.sqrt(1 + 8 * rdm.size)) / 2
            if self.n_cond % 1 != 0:
                raise NameError(
                    "RDM vector needs to have size of ncond*(ncond-1)/2")
            self.rdm = rdm
        else:  # User passed a matrix
            self.rdm_obj = RDMs(np.array([rdm]))
            self.rdm, _, self.n_cond = batch_to_vectors(np.array([rdm]))
            self.rdm = self.rdm[0]
        self.n_param = 0
        self.default_fitter = fit_mock
        self.rdm_obj.pattern_descriptors['index'] = np.arange(self.n_cond)

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
            rsatoolbox.rdm.RDMs: rdm object

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
            self.n_cond = rdm.n_cond
        elif rdm.ndim == 2:  # User supplied vectors
            self.rdm_obj = RDMs(rdm)
            self.n_cond = (1 + np.sqrt(1 + 8 * rdm.shape[1])) / 2
            if self.n_cond % 1 != 0:
                raise NameError(
                    "RDM vector needs to have size of ncond*(ncond-1)/2")
            self.rdm = rdm
        else:  # User passed matrixes
            self.rdm_obj = RDMs(rdm)
            self.rdm, _, self.n_cond = batch_to_vectors(rdm)
        self.n_param = 1
        self.n_rdm = self.rdm_obj.n_rdm
        self.default_fitter = fit_select

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
            rsatoolbox.rdm.RDMs: rdm object

        """
        return self.rdm_obj[theta]


class ModelWeighted(Model):
    """
    weighted Model
    models the RDM as a weighted sum of a set of RDMs
    """

    # Model Constructor
    def __init__(self, name, rdm):
        Model.__init__(self, name)
        if isinstance(rdm, RDMs):
            self.rdm_obj = rdm
            self.rdm = rdm.get_vectors()
            self.n_cond = rdm.n_cond
        elif rdm.ndim == 2:  # User supplied vectors
            self.rdm_obj = RDMs(rdm)
            self.n_cond = (1 + np.sqrt(1 + 8 * rdm.shape[1])) / 2
            if self.n_cond % 1 != 0:
                raise NameError(
                    "RDM vector needs to have size of ncond*(ncond-1)/2")
            self.rdm = rdm
        else:  # User passed matrixes
            self.rdm_obj = RDMs(rdm)
            self.rdm, _, self.n_cond = batch_to_vectors(rdm)
        self.n_param = self.rdm_obj.n_rdm
        self.n_rdm = self.rdm_obj.n_rdm
        self.default_fitter = fit_optimize

    def predict(self, theta=None):
        """ Returns the predicted rdm vector

        theta are the weights for the different rdms

        Args:
            theta(numpy.ndarray): the model parameter vector (one dimensional)

        Returns:
            rdm vector

        """
        if theta is None:
            theta = np.ones(self.n_rdm)
        theta = np.array(theta)
        return np.matmul(self.rdm.T, theta.reshape(-1))

    def predict_rdm(self, theta=None):
        """ Returns the predicted rdm vector

        For the fixed model there are no parameters.

        Args:
            theta(numpy.ndarray): the model parameter vector (one dimensional)

        Returns:
            rsatoolbox.rdm.RDMs: rdm object

        """
        if theta is None:
            theta = np.ones(self.n_rdm)
        theta = np.array(theta)
        dissimilarities = np.matmul(self.rdm.T, theta.reshape(-1))
        rdms = RDMs(
            dissimilarities.reshape(1, -1),
            dissimilarity_measure=self.rdm_obj.dissimilarity_measure,
            descriptors=self.rdm_obj.descriptors,
            pattern_descriptors=self.rdm_obj.pattern_descriptors)
        return rdms


class ModelInterpolate(Model):
    """
    inpterpolation Model
    models the RDM as an interpolation between 2 neigboring rdms
    """

    # Model Constructor
    def __init__(self, name, rdm):
        Model.__init__(self, name)
        if isinstance(rdm, RDMs):
            self.rdm_obj = rdm
            self.rdm = rdm.get_vectors()
            self.n_cond = rdm.n_cond
        elif rdm.ndim == 2:  # User supplied vectors
            self.rdm_obj = RDMs(rdm)
            self.n_cond = (1 + np.sqrt(1 + 8 * rdm.shape[1])) / 2
            if self.n_cond % 1 != 0:
                raise NameError(
                    "RDM vector needs to have size of ncond*(ncond-1)/2")
            self.rdm = rdm
        else:  # User passed matrixes
            self.rdm_obj = RDMs(rdm)
            self.rdm, _, self.n_cond = batch_to_vectors(rdm)
        self.n_param = self.rdm_obj.n_rdm
        self.n_rdm = self.rdm_obj.n_rdm
        self.default_fitter = fit_interpolate

    def predict(self, theta=None):
        """ Returns the predicted rdm vector

        theta are the weights for the different rdms

        Args:
            theta(numpy.ndarray): the model parameter vector (one dimensional)

        Returns:
            rdm vector

        """
        if theta is None:
            theta = np.zeros(self.n_rdm)
            theta[0] = 0.5
            theta[1] = 0.5
        theta = np.array(theta)
        return np.matmul(self.rdm.T, theta.reshape(-1))

    def predict_rdm(self, theta=None):
        """ Returns the predicted rdm vector

        For the fixed model there are no parameters.

        Args:
            theta(numpy.ndarray): the model parameter vector (one dimensional)

        Returns:
            rsatoolbox.rdm.RDMs: rdm object

        """
        if theta is None:
            theta = np.ones(self.n_rdm)
        theta = np.maximum(theta, 0)
        theta = np.array(theta)
        dissimilarities = np.matmul(self.rdm.T, theta.reshape(-1))
        rdms = RDMs(
            dissimilarities.reshape(1, -1),
            dissimilarity_measure=self.rdm_obj.dissimilarity_measure,
            descriptors=self.rdm_obj.descriptors,
            pattern_descriptors=self.rdm_obj.pattern_descriptors)
        return rdms


def model_from_dict(model_dict):
    """ recreates a model object from a dictionary

    Args:
        model_dict(dict): The dictionary to be turned into a model

    Returns
        model(Model): The recreated model

    """
    if model_dict['rdm']:
        rdm_obj = rdms_from_dict(model_dict['rdm'])
    if model_dict['type'] == 'Model':
        model = Model(model_dict['name'])
    elif model_dict['type'] == 'ModelFixed':
        model = ModelFixed(model_dict['name'], rdm_obj)
    elif model_dict['type'] == 'ModelSelect':
        model = ModelSelect(model_dict['name'], rdm_obj)
    elif model_dict['type'] == 'ModelWeighted':
        model = ModelWeighted(model_dict['name'], rdm_obj)
    elif model_dict['type'] == 'ModelInterpolate':
        model = ModelInterpolate(model_dict['name'], rdm_obj)
    return model
