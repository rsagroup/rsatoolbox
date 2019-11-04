#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 10:17:44 2019

@author: heiko
"""

import numpy as np
from pyrsa.rdm.rdms import RDMs
from pyrsa.data.dataset import Dataset


def calc_rdm(dataset, method = 'Euclidean',descriptor = None, noise = None):
    """
    calculates an RDM from an input dataset

        Args:
            dataset (pyrsa.data.DatasetBase):
                The dataset the RDM is computed from
            method (String):
                a description of the dissimilarity measure (e.g. 'Euclidean')
            descriptor (String):
                obs_descriptors used to define the rows/columns of the RDM
            noise (numpy.ndarray):
                precision matrix used to calculate the RDM
                used only for Mahalanobis and Crossnobis estimators
        Returns:
            RDMs object with the one RDM
    """
    measurements = dataset.measurements
    
    if method == 'Mahalanobis':
        if noise is None:
            noise = np.eye(measurements.shape[-1])
        rdm = calc_rdm_mahalanobis(dataset,noise)
    elif method == 'Euclidean':
        rdm = calc_rdm_euclid(dataset)
    elif method == 'Crossnobis':
        if noise is None:
            noise = np.eye(measurements.shape[-1])
        rdm = calc_rdm_crossnobis(measurements, noise)
    else:
        raise(NotImplementedError)
    return rdm

def calc_rdm_euclid(dataset,descriptor=None):
    """
    calculates an RDM from an input dataset using euclidean distance
    If multiple instances of the same condition are found in the dataset
    they are averaged.

        Args:
            dataset (pyrsa.data.DatasetBase):
                The dataset the RDM is computed from
            descriptor (String):
                obs_descriptors used to define the rows/columns of the RDM
                defaults to one row/column per row in the dataset
        Returns:
            RDMs object with the one RDM
    """
    measurements = dataset.measurements
    shape = measurements.shape[0:-2]
    RDM = np.zeros(shape+(measurements.shape[-2],measurements.shape[-2]))
    if len(RDM.shape)==2:
        for i in range(RDM.shape[0]):
            for j in range(i,RDM.shape[1]):
                RDM[i,j] = np.matmul((measurements[i,:]-measurements[j,:]),(measurements[i,:]-measurements[j,:])) /measurements.shape[1]
                RDM[j,i] = RDM[i,j]
    else:
        RDMv = RDM.reshape((np.prod(shape),measurements.shape[-2],measurements.shape[-2]))
        measurements = measurements.reshape((np.prod(shape),measurements.shape[-2],measurements.shape[-1]))
        for iRDM in range(RDMv.shape[0]):
            for i in range(RDM.shape[-2]):
                for j in range(i,RDM.shape[-1]):
                    RDMv[iRDM,i,j] = np.matmul((measurements[iRDM,i,:]-measurements[iRDM,j,:]),(measurements[iRDM,i,:]-measurements[iRDM,j,:])) /measurements.shape[-1]
                    RDMv[iRDM,j,i] = RDMv[iRDM,i,j]
    rdm = RDMs(dissimilarities = None, dissimilarity_measure = 'euclidean',
                 descriptors = dataset.descriptors)
    return rdm


def calc_rdm_mahalanobis(dataset, descriptor=None, noise=None):
    """
    calculates an RDM from an input dataset using mahalanobis distance
    If multiple instances of the same condition are found in the dataset
    they are averaged.

        Args:
            dataset (pyrsa.data.DatasetBase):
                The dataset the RDM is computed from
            descriptor (String):
                obs_descriptors used to define the rows/columns of the RDM
                defaults to one row/column per row in the dataset
            noise (numpy.ndarray):
                precision matrix used to calculate the RDM
        Returns:
            RDMs object with the one RDM
    """
    measurements = np.array(measurements)
    shape = measurements.shape[0:-2]
    RDM = np.zeros(shape+(measurements.shape[-2],measurements.shape[-2]))
    if len(RDM.shape)==2:
        for i in range(RDM.shape[0]):
            for j in range(i,RDM.shape[1]):
                RDM[i,j] = np.matmul((measurements[i,:]-measurements[j,:]),np.linalg.solve(noise,(measurements[i,:]-measurements[j,:]))) /measurements.shape[1]
                RDM[j,i] = RDM[i,j]
    else:
        RDMv = RDM.reshape((np.prod(shape),measurements.shape[-2],measurements.shape[-2]))
        measurements = measurements.reshape((np.prod(shape),measurements.shape[-2],measurements.shape[-1]))
        for iRDM in range(RDMv.shape[0]):
            for i in range(RDMv.shape[1]):
                for j in range(i,RDMv.shape[2]):
                    RDMv[iRDM,i,j] = np.matmul((measurements[i,:]-measurements[j,:]),np.linalg.solve(noise,(measurements[i,:]-measurements[j,:]))) /measurements.shape[-1]
                    RDMv[iRDM,j,i] = RDM[iRDM,i,j]
    return RDM


def calc_rdm_crossnobis(dataset, descriptor=None, noise= None, Nfolds=None):
    """
    calculates an RDM from an input dataset using Cross-nobis distance
    If multiple instances of the same condition are found in the dataset
    they are averaged.

        Args:
            dataset (pyrsa.data.DatasetBase):
                The dataset the RDM is computed from
            descriptor (String):
                obs_descriptors used to define the rows/columns of the RDM
                defaults to one row/column per row in the dataset
            noise (numpy.ndarray):
                precision matrix used to calculate the RDM
        Returns:
            RDMs object with the one RDM
    """
    # measurements should be a list of measurements computed for the different repetitions/folds
    if Nfolds is None:
        Nfolds = len(measurements) # default to leave one out crossvalidation
    measurements = np.array(measurements)
    shape = measurements.shape[0:-3]
    RDMs = np.zeros(shape+(Nfolds,measurements.shape[-2],measurements.shape[-2]))
    if len(shape)==0:
        for iCross in range(Nfolds):
            measurements1 = measurements[iCross]
            measurements2 = np.mean(np.concatenate((measurements[:iCross],measurements[(iCross+1):]),axis=0),axis=0)
            RDMs[iCross] = calc_RDM_crossnobis_single(measurements1,measurements2,noise)
    else:
        RDMv = RDMs.reshape((np.prod(shape),Nfolds,measurements.shape[-2],measurements.shape[-2]))
        measurements = measurements.reshape((np.prod(shape),measurements.shape[-3],measurements.shape[-2],measurements.shape[-1]))
        for iRDM in range(RDMv.shape[0]):
            for iCross in range(Nfolds):
                measurements1 = measurements[iRDM,iCross]
                measurements2 = np.mean(np.concatenate((measurements[iRDM,:iCross],measurements[iRDM,(iCross+1):]),axis=0),axis=0)
                RDMs[iCross] = calc_RDM_crossnobis_single(measurements1,measurements2,noise)
    RDM = np.mean(RDMs,axis=-3)
    return RDM
    

def calc_RDM_crossnobis_single(measurements1,measurements2,noise):
    C = get_cotrast_matrix(measurements1.shape[0])
    Ds1 = np.matmul(C,measurements1)
    Ds2 = np.matmul(C,measurements2)
    Ds2 = np.linalg.solve(noise,Ds2.transpose())
    RDM = np.einsum('kj,jk->k',Ds1,Ds2)/measurements1.shape[1]
    return get_rdm_matrix(RDM)