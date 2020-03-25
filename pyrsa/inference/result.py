#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 14:42:47 2020

@author: heiko
"""

import numpy as np
import h5py
import os
import pyrsa.model

class Result:
    """ Result class storing results for a set of models with the models,
    the results matrix and the noise ceiling

    Args:
        models(list of pyrsa.model.Model):
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
    def __init__(self, models, evaluations, method, cv_method, noise_ceiling):
        if isinstance(models, pyrsa.model.Model):
            models = [models]
        assert len(models) == evaluations.shape[1], 'evaluations shape does' \
            + 'not match number of models'
        self.models = models
        self.n_model = len(models)
        self.evaluations = evaluations
        self.method = method
        self.cv_method = cv_method
        self.noise_ceiling = noise_ceiling

    def save(self, filename):
        """ saves the results into a file. 

        the list of models is currently not saved!

        Args:
            filename(String): path to the filelocation

        """
        if isinstance(filename, str) and os.path.isfile(filename):
            raise FileExistsError('File already exists')
        file = h5py.File(filename, 'a')
        file['evaluations'] = self.evaluations
        file['noise_ceiling'] = self.noise_ceiling
        file.attrs['method'] = self.method
        file.attrs['cv_method'] = self.cv_method
        file.attrs['version'] = 3


def load_results(filename):
    """ loads a Result object from disc

    Args:
        filename(String): path to the filelocation

    """
    file = h5py.File(filename, 'a')
    evaluations = np.array(file['evaluations'])
    models = [None] * evaluations.shape[1]
    noise_ceiling = np.array(file['noise_ceiling'])
    method = file.attrs['method']
    cv_method = file.attrs['cv_method']
    return Result(models, evaluations, method, cv_method, noise_ceiling)