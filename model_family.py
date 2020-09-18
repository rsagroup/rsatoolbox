#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Definition of RSA Model family
"""

import numpy as np
import itertools

class ModelFamily():
    """
    Model Family class.
    Defines members that every class needs to have, but does not implement any
    interesting behavior. Inherit from this class to define specific model
    types
    """
    def __init__(self, models):
        """
        initializes the model family class
        Args:
            models(List of pyrsa.model.models): List of models
        """
        self.num_models = len(models)
        self.family, self.indices = self.__create_model_family(models)


    def __create_model_family(self,models):
        """ Returns the model family and

        For the fixed model there are no parameters. theta is ignored.

        Args:
            theta(numpy.ndarray): the model parameter vector (one dimensional)

        Returns:
            rdm vector

        """
        model_family = []
        indices = []
        possible_models = range(self.num_models)
        for L in range(1, self.num_models+1):
            for subset in itertools.combinations(models, L):
                print(len(subset))
                model_family.append(subset)
        return model_family
        return family, indices
