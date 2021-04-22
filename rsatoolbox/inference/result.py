#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Result object definition
"""

import numpy as np
import rsatoolbox.model
from rsatoolbox.util.file_io import write_dict_hdf5
from rsatoolbox.util.file_io import write_dict_pkl
from rsatoolbox.util.file_io import read_dict_hdf5
from rsatoolbox.util.file_io import read_dict_pkl
from rsatoolbox.util.file_io import remove_file
from rsatoolbox.util.inference_util import extract_variances


class Result:
    """ Result class storing results for a set of models with the models,
    the results matrix and the noise ceiling

    Args:
        models(list of rsatoolbox.model.Model):
            the evaluated models
        evaluations(numpy.ndarray):
            evaluations of the models over bootstrap/crossvalidation
            format: bootstrap_samples x models x crossval & others
            such that np.mean(evaluations[i,j]) is a valid evaluation for the
            jth model on the ith bootstrap-sample
        method(String):
            the evaluation method
        cv_method(String):
            crossvalidation specification
        noise_ceiling(numpy.ndarray):
            noise ceiling such that np.mean(noise_ceiling[0]) is the lower
            bound and np.mean(noise_ceiling[1]) is the higher one.

    Attributes:
        as inputs

    """

    def __init__(self, models, evaluations, method, cv_method, noise_ceiling,
                 variances=None, dof=1):
        if isinstance(models, rsatoolbox.model.Model):
            models = [models]
        assert len(models) == evaluations.shape[1], 'evaluations shape does' \
            + 'not match number of models'
        self.models = models
        self.n_model = len(models)
        self.evaluations = np.array(evaluations)
        self.method = method
        self.cv_method = cv_method
        self.noise_ceiling = np.array(noise_ceiling)
        self.variances = variances
        self.dof = dof
        if variances is not None:
            # if the variances only refer to the models this should have the
            # same number of entries as the models list.
            if variances.ndim == 0:
                nc_included = False
            else:
                nc_included = variances.shape[-1] != len(models)
            self.model_var, self.diff_var, self.noise_ceil_var = \
                extract_variances(variances, nc_included)
        else:
            self.model_var = None
            self.diff_var = None
            self.noise_ceil_var = None

    def save(self, filename, file_type='hdf5', overwrite=False):
        """ saves the results into a file.

        Args:
            filename(String): path to the file
                [or opened file]
            file_type(String): Type of file to create:
                hdf5: hdf5 file
                pkl: pickle file
            overwrite(Boolean): overwrites file if it already exists

        """
        result_dict = self.to_dict()
        if overwrite:
            remove_file(filename)
        if file_type == 'hdf5':
            write_dict_hdf5(filename, result_dict)
        elif file_type == 'pkl':
            write_dict_pkl(filename, result_dict)

    def to_dict(self):
        """ Converts the RDMs object into a dict, which can be used for saving

        Returns:
            results_dict(dict): A dictionary with all the information needed
                to regenerate the object

        """
        result_dict = {}
        result_dict['evaluations'] = self.evaluations
        result_dict['dof'] = self.dof
        result_dict['variances'] = self.variances
        result_dict['noise_ceiling'] = self.noise_ceiling
        result_dict['method'] = self.method
        result_dict['cv_method'] = self.cv_method
        result_dict['models'] = {}
        for i_model in range(len(self.models)):
            key = 'model_%d' % i_model
            result_dict['models'][key] = self.models[i_model].to_dict()
        return result_dict


def load_results(filename, file_type=None):
    """ loads a Result object from disc

    Args:
        filename(String): path to the filelocation

    """
    if file_type is None:
        if isinstance(filename, str):
            if filename[-4:] == '.pkl':
                file_type = 'pkl'
            elif filename[-3:] == '.h5' or filename[-4:] == 'hdf5':
                file_type = 'hdf5'
    if file_type == 'hdf5':
        data_dict = read_dict_hdf5(filename)
    elif file_type == 'pkl':
        data_dict = read_dict_pkl(filename)
    else:
        raise ValueError('filetype not understood')
    return result_from_dict(data_dict)


def result_from_dict(result_dict):
    """ recreate Results object from dictionary

    Args:
        result_dict(dict): dictionary to regenerate

    Returns:
        result(Result): the recreated object

    """
    if 'variances' in result_dict.keys():
        variances = result_dict['variances']
    else:
        variances = None
    if 'dof' in result_dict.keys():
        dof = result_dict['dof']
    else:
        dof = None
    evaluations = result_dict['evaluations']
    method = result_dict['method']
    cv_method = result_dict['cv_method']
    noise_ceiling = result_dict['noise_ceiling']
    models = [None] * len(result_dict['models'])
    for i_model in range(len(result_dict['models'])):
        key = 'model_%d' % i_model
        models[i_model] = rsatoolbox.model.model_from_dict(
            result_dict['models'][key])
    return Result(models, evaluations, method, cv_method, noise_ceiling,
                  variances=variances, dof=dof)
