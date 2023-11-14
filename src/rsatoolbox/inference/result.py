#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Result object definition
"""

import numpy as np
import scipy.stats
import rsatoolbox.model
from rsatoolbox.io.hdf5 import read_dict_hdf5, write_dict_hdf5
from rsatoolbox.io.pkl import read_dict_pkl, write_dict_pkl
from rsatoolbox.util.file_io import remove_file
from rsatoolbox.util.inference_util import extract_variances
from rsatoolbox.util.inference_util import all_tests, pair_tests, nc_tests, zero_tests


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
                 variances=None, dof=1, fitter=None, n_rdm=None, n_pattern=None):
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
        self.fitter = fitter
        self.n_bootstraps = evaluations.shape[0]
        self.n_rdm = n_rdm
        self.n_pattern = n_pattern
        if variances is not None:
            # if the variances only refer to the models this should have the
            # same number of entries as the models list.
            if variances.ndim == 0:
                nc_included = False
            else:
                nc_included = variances.shape[-1] != len(models)
            self.model_var, self.diff_var, self.noise_ceil_var = \
                extract_variances(variances, nc_included, n_rdm, n_pattern)
        else:
            self.model_var = None
            self.diff_var = None
            self.noise_ceil_var = None

    def __repr__(self):
        """ defines string which is printed for the object
        """
        return (f'rsatoolbox.inference.Result\n'
                f'containing evaluations for {self.n_model} models\n'
                f'evaluated using {self.cv_method} of {self.method}'
                )

    def __str__(self):
        """ defines the output of print
        """
        return self.summary()

    def summary(self, test_type='t-test'):
        """
        Human readable summary of the results

        Args:
            test_type(String):
                What kind of tests to run.
                See rsatoolbox.util.inference_util.all_tests for options
        """
        summary = f'Results for running {self.cv_method} evaluation for {self.method} '
        summary += f'on {self.n_model} models:\n\n'
        name_length = max([max(len(m.name) for m in self.models) + 1, 6])
        means = self.get_means()
        sems = self.get_sem()
        if means is None:
            means = np.nan * np.ones(self.n_model)
        if sems is None:
            sems = np.nan * np.ones(self.n_model)
        try:
            p_zero = self.test_zero(test_type=test_type)
            p_noise = self.test_noise(test_type=test_type)
        except ValueError:
            p_zero = np.nan * np.ones(self.n_model)
            p_noise = np.nan * np.ones(self.n_model)
        # header of the results table
        summary += 'Model' + (' ' * (name_length - 5))
        summary += '|   Eval \u00B1 SEM   |'
        summary += ' p (against 0) |'
        summary += ' p (against NC) |\n'
        summary += '-' * (name_length + 51)
        summary += '\n'
        for i, m in enumerate(self.models):
            summary += m.name + (' ' * (name_length - len(m.name)))
            summary += f'| {means[i]: 5.3f} \u00B1 {sems[i]:4.3f} |'
            if p_zero[i] < 0.001:
                summary += '      < 0.001  |'
            else:
                summary += f'{p_zero[i]:>13.3f}  |'
            if p_noise[i] < 0.001:
                summary += '       < 0.001  |'
            else:
                summary += f'{p_noise[i]:>14.3f}  |'
            summary += '\n'
        summary += '\n'
        if self.cv_method == 'crossvalidation':
            summary += 'No p-values available as crossvalidation provides no variance estimate'
        elif test_type == 't-test':
            summary += 'p-values are based on uncorrected t-tests'
        elif test_type == 'bootstrap':
            summary += 'p-values are based on percentiles of the bootstrap samples'
        elif test_type == 'ranksum':
            summary += 'p-values are based on ranksum tests'
        return summary

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
        result_dict['n_rdm'] = self.n_rdm
        result_dict['n_pattern'] = self.n_pattern
        result_dict['models'] = {}
        for i_model in range(len(self.models)):
            key = 'model_%d' % i_model
            result_dict['models'][key] = self.models[i_model].to_dict()
        return result_dict

    def test_all(self, test_type='t-test'):
        """ returns all p-values: p_pairwise, p_zero & p_noise

        Args:
            test_type(String):
                What kind of tests to run.
                See rsatoolbox.util.inference_util.all_tests for options
        """
        p_pairwise, p_zero, p_noise = all_tests(
            self.evaluations, self.noise_ceiling, test_type,
            model_var=self.model_var, diff_var=self.diff_var,
            noise_ceil_var=self.noise_ceil_var, dof=self.dof)
        return p_pairwise, p_zero, p_noise

    def test_pairwise(self, test_type='t-test'):
        """returns the pairwise test p-values """
        return pair_tests(self.evaluations, test_type, self.diff_var, self.dof)

    def test_zero(self, test_type='t-test'):
        """returns the p-values for the tests against 0 """
        return zero_tests(self.evaluations, test_type, self.model_var, self.dof)

    def test_noise(self, test_type='t-test'):
        """returns the p-values for the tests against the noise ceiling"""
        return nc_tests(self.evaluations, self.noise_ceiling,
                        test_type, self.noise_ceil_var, self.dof)

    def get_means(self):
        """ returns the mean evaluations per model """
        if self.cv_method == 'fixed':
            perf = np.mean(self.evaluations, axis=0)
            perf = np.nanmean(perf, axis=-1)
        elif self.cv_method == 'crossvalidation':
            perf = np.mean(self.evaluations, axis=0)
            perf = np.nanmean(perf, axis=-1)
        else:
            perf = self.evaluations
            while len(perf.shape) > 2:
                perf = np.nanmean(perf, axis=-1)
            perf = perf[~np.isnan(perf[:, 0])]
            perf = np.mean(perf, axis=0)
        return perf

    def get_sem(self):
        """ returns the SEM of the evaluation per model """
        if self.model_var is None:
            return None
        return np.sqrt(np.maximum(self.model_var, 0))

    def get_ci(self, ci_percent, test_type='t-test'):
        """ returns confidence intervals for the evaluations"""
        prop_cut = (1 - ci_percent) / 2
        if test_type == 'bootstrap':
            perf = self.evaluations
            while len(perf.shape) > 2:
                perf = np.nanmean(perf, axis=-1)
            framed_evals = np.concatenate(
                (np.tile(np.array(([-np.inf], [np.inf])),
                         (1, self.n_model)),
                 perf),
                axis=0)
            ci = [np.quantile(framed_evals, prop_cut, axis=0),
                  np.quantile(framed_evals, 1 - prop_cut, axis=0)]
        else:
            tdist = scipy.stats.t
            std_eval = self.get_sem()
            means = self.get_means()
            ci = [means + std_eval * tdist.ppf(prop_cut, self.dof),
                  means - std_eval * tdist.ppf(prop_cut, self.dof)]
        return ci

    def get_errorbars(self, eb_type='sem', test_type='t-test'):
        """ returns errorbars for the model evaluations"""
        if eb_type.lower() == 'sem':
            errorbar_low = self.get_sem()
            errorbar_high = errorbar_low
        elif eb_type[0:2].lower() == 'ci':
            if len(eb_type) == 2:
                ci_percent = 0.95
            else:
                ci_percent = float(eb_type[2:]) / 100
            ci = self.get_ci(ci_percent, test_type)
            means = self.get_means()
            errorbar_low = means - ci[0]
            errorbar_high = ci[1] - means
            limits = np.concatenate((errorbar_low, errorbar_high))
            if np.isnan(limits).any() or (abs(limits) == np.inf).any():
                raise ValueError(
                    'plot_model_comparison: Too few bootstrap samples for ' +
                    'the requested confidence interval: ' + eb_type + '.')
        return (errorbar_low, errorbar_high)

    def get_model_var(self):
        """ returns the variance of the evaluation per model """
        return self.model_var

    def get_noise_ceil(self):
        """ returns the noise ceiling for the model evaluations """
        return self.noise_ceiling


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
    n_rdm = result_dict['n_rdm']
    n_pattern = result_dict['n_pattern']
    return Result(models, evaluations, method, cv_method, noise_ceiling,
                  variances=variances, dof=dof, n_rdm=n_rdm, n_pattern=n_pattern)
