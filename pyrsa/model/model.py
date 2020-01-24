#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Definition of RSA Model class and subclasses
@author: jdiedrichsen, heiko
"""

import numpy as np
from pyrsa.rdm import RDMs
from pyrsa.rdm import compare
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
        self.default_fitter = fit_mock

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

    def fit(self, data):
        """ fit the model to a RDM object data

        Args:
            data(RDM object): the RDMs to be fit with the model

        Returns:
            theta(numpy.ndarray): parameter vector (one dimensional)
        """
        return self.default_fitter(self, data, method='cosine',
                                   pattern_sample=None,
                                   pattern_descriptor=None)


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
            self.rdm = batch_to_vectors(np.array([rdm]))[0]
            self.n_cond = self.rdm_obj.n_cond
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
            self.rdm_obj = RDMs(rdm)
            self.n_cond = (1 + np.sqrt(1 + 8 * rdm.shape[1])) / 2
            if self.n_cond % 1 != 0:
                raise NameError(
                    "RDM vector needs to have size of ncond*(ncond-1)/2")
            self.rdm = rdm
        else:  # User passed matrixes
            self.rdm_obj = RDMs(rdm)
            self.rdm = batch_to_vectors(rdm)
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
            pyrsa.rdm.RDMs: rdm object

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
        elif rdm.ndim == 2:  # User supplied vectors
            self.rdm_obj = RDMs(rdm)
            self.n_cond = (1 + np.sqrt(1 + 8 * rdm.shape[1])) / 2
            if self.n_cond % 1 != 0:
                raise NameError(
                    "RDM vector needs to have size of ncond*(ncond-1)/2")
            self.rdm = rdm
        else:  # User passed matrixes
            self.rdm_obj = RDMs(rdm)
            self.rdm = batch_to_vectors(rdm)
        self.n_param = self.rdm_obj.n_rdm
        self.n_rdm = self.rdm_obj.n_rdm
        self.default_fitter = fit_select

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
            pyrsa.rdm.RDMs: rdm object

        """
        if theta is None:
            theta = np.ones(self.n_rdm)
        theta = np.maximum(theta, 0)
        theta = np.array(theta)
        dissimilarities = np.matmul(self.rdm.T, theta.reshape(-1))
        rdms = RDMs(dissimilarities.reshape(1,-1),
                 dissimilarity_measure=self.rdm_obj.dissimilarity_measure,
                 descriptors=self.rdm_obj.descriptors,
                 pattern_descriptors=self.rdm_obj.pattern_descriptors)
        return rdms


def fit_mock(model, data, method='cosine', pattern_sample=None,
             pattern_descriptor=None):
    """ formally acceptable fitting method which always returns a vector of
    zeros

    Args:
        model(pyrsa.model.Model): model to be fit
        data(pyrsa.rdm.RDMs): Data to fit to
        method(String): Evaluation method
        pattern_sample(numpy.ndarray): Which patterns are sampled
        pattern_descriptor(String): Which descriptor is used

    Returns:
        theta(numpy.ndarray): parameter vector

    """
    return np.zeros(model.n_param)


def fit_select(model, data, method='cosine', pattern_sample=None,
               pattern_descriptor=None):
    """ fits selection models by evaluating each rdm and selcting the one
    with best performance. Works only for ModelSelect

    Args:
        model(pyrsa.model.Model): model to be fit
        data(pyrsa.rdm.RDMs): Data to fit to
        method(String): Evaluation method
        pattern_sample(numpy.ndarray): Which patterns are sampled
        pattern_descriptor(String): Which descriptor is used

    Returns:
        theta(int): parameter vector

    """
    assert isinstance(model, ModelSelect)
    evaluations = np.zeros(model.n_rdm)
    for i_rdm in range(model.n_rdm):
        pred = model.predict_rdm(i_rdm)
        if not (pattern_sample is None or pattern_descriptor is None):
            pred = pred.subsample_pattern(pattern_descriptor, pattern_sample)
        evaluations[i_rdm] = np.mean(compare(pred, data, method=method))
    print(evaluations)
    theta = np.argmin(evaluations)
    return theta
