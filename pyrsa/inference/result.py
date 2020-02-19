#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 14:42:47 2020

@author: heiko
"""

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
