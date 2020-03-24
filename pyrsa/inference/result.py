#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 14:42:47 2020

@author: heiko
"""

import numpy as np
import pickle
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
        """ saves the evaluations into a file. This will create 3 files:
            [filename].npy: the model evaluations
            [filename]_ns.npy: the noise_ceiling
            [filename]_desc.pkl: the descriptors about the results
                                 (method, cv_method)

        the list of models is currently not saved!

        Args:
            filename(String): path to the filelocation

        """
        np.save(filename + '.npy', self.evaluations)
        np.save(filename + '_ns.npy', self.noise_ceiling)
        pickle.dump([self.method,
                     self.cv_method],
                    open(filename + '_desc.pkl','wb'))


def load_results(filename):
    """ loads a Result object from disc

    Args:
        filename(String): path to the filelocation

    """
    evaluations = np.load(filename + '.npy')
    models = [None] * evaluations.shape[1]
    noise_ceiling = np.load(filename + '_ns.npy')
    desc = pickle.load(open(filename + '_desc.pkl', 'rb'))
    return Result(models, evaluations, desc[0], desc[1], noise_ceiling)