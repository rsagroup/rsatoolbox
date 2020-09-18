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
        self.models = models
        self.num_family_members = 2**self.num_models-1
        self.family_list, self.model_indices = self.__create_model_family()

    def __create_model_family(self):
        """ Returns the model family and

        For the fixed model there are no parameters. theta is ignored.

        Args:
            theta(numpy.ndarray): the model parameter vector (one dimensional)

        Returns:
            rdm vector

        """
        family_list = []
        indices = np.zeros((self.num_family_members, self.num_models))
        models_index_array = range(self.num_models)
        count = 0
        for L in range(1, self.num_models+1):
            for subset in itertools.combinations(models_index_array, L):
                selected_indices = [models_index_array.index(x) for x in subset]
                indices[count,selected_indices] = 1
                family_list.append(subset)
                count+=1
        return family_list, indices

    def get_family_member(self,family_index):
        """
        """
        member_indices = list(self.family_list[family_index])
        family_member = [ self.models[i] for i in member_indices]

        return family_member

random_list = [4,6,5,2]
model_family = ModelFamily(random_list)
print(model_family.family_list)
print(model_family.model_indices)
for i in range(model_family.num_family_members):
    print(model_family.get_family_member(i))
