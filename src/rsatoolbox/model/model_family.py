#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Definition of RSA Model family
"""
import itertools
import numpy as np
from rsatoolbox.rdm import RDMs
from .model import ModelWeighted


class ModelFamily():
    """Short summary.

    Parameters
    ----------
    models : type
        Description of parameter `models`.

    Attributes
    ----------
    num_models : int
        total number of fixed models.
    num_family_members : int
        total number of possible combinations of fixed models 2**num_models.
    family_list : list of array
        list of selected fixed model indices for all family members.
    model_indices : numpy array
        binary array indicating the selected fixed models for each family member
    __create_model_family : method
        generates family_list and model_indices
    models

    """

    def __init__(self, models):
        """Class initialization.

        Parameters
        ----------
        models : list of fixed models
            List of models
        Returns
        -------
        None

        """
        self.num_models = len(models)
        self.models = models
        self.num_family_members = 2**self.num_models-1
        self.family_list, self.model_indices = self.__create_model_family()

    def __create_model_family(self):
        """Creates model family.

        Returns
        -------
        family_list : list of array
            list of selected fixed model indices for all family members.
        model_indices : numpy array
            binary array indicating the selected fixed models for each family member

        """
        family_list = []
        indices = np.zeros((self.num_family_members, self.num_models))
        models_index_array = range(self.num_models)
        count = 0
        for subset_length in range(1, self.num_models+1):
            for subset in itertools.combinations(models_index_array, subset_length):
                selected_indices = [models_index_array.index(x) for x in subset]
                indices[count, selected_indices] = 1
                family_list.append(subset)
                count += 1
        return family_list, indices

    def get_family_member(self, family_index):
        """returns family member given an input index

        Parameters
        ----------
        family_index : int
            Index corresponding to a family member

        Returns
        -------
        weighted_model
            family member corresponding to input index

        """

        member_indices = list(self.family_list[family_index])
        family_member = [self.models[i] for i in member_indices]

        return family_member

    def get_all_family_members(self):
        """returns a list of weighted models.

        Returns
        -------
        list
            list of weighted models.

        """
        all_family_members = []
        for family_index in range(self.num_family_members):
            family_member = self.get_family_member(family_index)
            member_rdms = []
            member_name = ""
            for model in family_member:
                member_rdms.append(model.predict_rdm().get_vectors().ravel())
                member_name = member_name + "_" + model.name
            member_rdms = np.array(member_rdms)
            member_rdms = RDMs(member_rdms)

            weighted_model = ModelWeighted(member_name, member_rdms)
            all_family_members.append(weighted_model)

        return all_family_members
