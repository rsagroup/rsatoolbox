#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Definition of RSA Model family

Authors: jdiedrichsen & kshitijd20
"""
import itertools
import numpy as np
import rsatoolbox as rsa
from rsatoolbox.rdm import RDMs
from .model import ModelWeighted, ModelFixed
import pandas as pd

class ModelFamily:
    """
    ModelFamily class is a list (iterable) of models,
    which is constructed from a combining a set of components in
    every possible way. Every components can be either switched in or out.
    You can specify a list of 'base components', which are always present.
    A Model family can be either constructed from a component model, or
    a list of (usually fixed) models.
    """
    def __init__(self,components, comp_names=None,base_components=None):
        """
        Args:
            components (list or RDMs object)
                A list of model RDMs, which are used to create the model family
                Alternative a simple stack of RDMs
            comp_name (list of str)
                A list of strings for each of the model components
            basecomponents (list or RDMs object)
                A list of RDM components that are always present (default none)
        """
        if type(components) is np.ndarray:
            components = RDMs(components)
        if type(components) is RDMs:
            self.n_comp = components.n_rdm
            self.comp_names = comp_names
            self.rdm = components.get_vectors()

        elif type(components) is list:
            raise(NameError('Needs to be still implemented'))
        else:
            raise(NameError('Input needs to be a list of models, ndarray, or a RDMs object'))

        # Check if component names are given:
        if self.comp_names is None:
            self.comp_names = [f'{d}' for d in np.arange(self.n_comp)+1]
        self.comp_names = np.array(self.comp_names)
        if self.n_comp > 12:
            raise(NameError('More than 12 components - a full search is probably not recommended '))
        else:
            self._create_model_family()

    def _create_model_family(self):
        """ Creates a model family for full model search

        comp_indices : numpy array
            binary array indicating the selected components for each family member

        """

        # Build all combination of 0,1,2... components
        self.n_models = 2 ** self.n_comp
        self.comp_indices = np.empty((self.n_models,0),dtype=int)

        ind = np.arange(self.n_models)
        for i in range(self.n_comp):
            self.comp_indices = np.c_[self.comp_indices,np.floor(ind/(2**i))%2]

        # Order the combinations by the number of components that they contain
        self.n_comp_per_m = self.comp_indices.sum(axis=1).astype(int)
        ind = np.argsort(self.n_comp_per_m)
        self.n_comp_per_m = self.n_comp_per_m[ind]
        self.comp_indices = self.comp_indices[ind,:]

        # Now build all model combinations as individual models
        self.models = []
        self.model_names = []
        for m in range(self.n_models):
            ind = self.comp_indices[m]>0
            if ind.sum()==0:
                name = 'base'
                mod = ModelFixed(name,np.zeros((self.rdm.shape[1],)))
            else:
                name = '+'.join(self.comp_names[ind])
                mod = ModelWeighted(name,self.rdm[ind,:])
                mod.default_fitter = rsa.model.fitter.fit_optimize_positive
            self.model_names.append(name)
            self.models.append(mod)

    def __getitem__(self,key):
        return self.models[key]

    def __len__(self):
        return self.n_models

    def get_layout(self):
        """generate 2d layout of the model tree
        root model will be at (0,0)
        """
        x = np.zeros((self.n_models,))
        y = np.zeros((self.n_models,))
        max_comp=np.max(self.n_comp_per_m)
        for i in range(max_comp+1):
            ind = self.n_comp_per_m==i
            y[ind]=i
            x_coord = np.arange(np.sum(ind))
            x[ind]= x_coord - x_coord.mean()
        return x,y

    def get_connectivity(self):
        """ return a connectivty
        matrix that determines whether
        2 models only differ by a single component
        """
        connect = np.zeros((self.n_models,self.n_models),dtype=int)
        connect_sgn = np.zeros((self.n_models,self.n_models),dtype=int)
        for i in range(self.n_comp):
            diff = self.comp_indices[:,i].reshape((-1,1)) - self.comp_indices[:,i].reshape((1,-1))
            connect = connect + np.abs(diff).astype(int)
            connect_sgn = connect_sgn + diff.astype(int)
        return (connect==1)*connect_sgn

    def model_posterior(self,result,method='AIC',format='Result'):
        """ Determine posterior of the model across model family

        Args:
            result (inference.Result):
                Result object from the evaluation of the model family
            method (string):
                Method by which to correct for number of parameters(k)
                'AIC' (default)
                None: No correction - use if crossvalidated likelihood is used
            format (string):
                Return format for posterior
                'ndarray': Simple N x n_models np.array
                'Results': Add to result class and return entire
        Returns:
            posterior (Result or ndarray):
                Model posterior - rows are data set, columns are models
        """
        if type(result) is not rsa.inference.Result:
            raise(NameError('Input needs to be result structure'))
        if result.n_model != self.n_models:
            raise(NameError('Number of models in result does not fit ModelFamily'))

        # Get relative log-liklihood of the distances under the model
        # For linear regression the log-liklihood is (up to a constant)
        # LL = -n/2*log(1-R^2)
        if result.method in ['cosine','Pearson']:
            R=result.evaluations.mean(axis=0)
            R[np.isnan(R)]=0
            R[R<0]=0
            n = self.rdm.shape[1]
            rLL = -n/2*np.log(1-R**2).T

        # Correct for model df or rely on cv
        if method=='AIC':
            crit = rLL - self.n_comp_per_m
        elif method is None:
            crit = rLL
        else:
            raise(NameError('Method needs be either AIC, or None'))

        # Safe transform into probability
        crit = crit - crit.max(axis=1).reshape(-1,1)
        crit = np.exp(crit)
        p = crit / crit.sum(axis=1).reshape(-1,1)

        if format == 'Results':
            result.posterior = p
            return result
        else:
            return p

    def component_posterior(self,result,method='AIC',format='DataFrame'):
        """ Determine the posterior of the component (absence / presence)

        Args:
            result (inference.Result):
                Result object from the evaluation of the model family
            method (string):
                Method by which to correct for number of parameters(k)
                'AIC' (default): LL-k
                None: No correction - use if crossvalidated likelihood is used
            format (string):
                Return format for posterior
                'ndarray': Simple N x n_models np.array
                'DataFrame': pandas Data frame

        Returns:
            posterior (DataFrame):
                Component posterior - rows are data set, columns are components
        """
        mposterior = self.model_posterior(result,method,format='ndarray')
        cposterior = np.empty((mposterior.shape[0],self.n_comp))

        for i in range(self.n_comp):
            cposterior[:,i] = mposterior[:,self.comp_indices[:,i]==1].sum(axis=1)

        if format == 'DataFrame':
            return pd.DataFrame(data=cposterior,
                        index=np.arange(cposterior.shape[0]),
                        columns = self.comp_names)

        return cposterior

    def component_bayesfactor(self,result,method='AIC',format='ndarray'):
        """ Returns a log-bayes factor for each component

        Args:
            result (inference.Result):
                Result object from the evaluation of the model family
            method (string):
                Method by which to correct for number of parameters(k)
                'AIC' (default): LL-k
                None: No correction - use if crossvalidated likelihood is used
            format (string):
                Return format for posterior
                'ndarray': Simple N x n_models np.array
                'DataFrame': pandas Data frame

        Returns:
            posterior (DataFrame):
                Component posterior - rows are data set, columns are components
        """
        mposterior = self.model_posterior(result,method)
        c_bf = np.empty((mposterior.shape[0],self.n_comp))

        for i in range(self.n_comp):
            c_bf[:,i] = np.log(mposterior[:,self.comp_indices[:,i]==1].sum(axis=1))-np.log(mposterior[:,self.comp_indices[:,i]==0].sum(axis=1))

        if format == 'DataFrame':
            return pd.DataFrame(data=c_bf,
                        index=np.arange(c_bf.shape[0]),
                        columns = self.comp_names)

        return c_bf


