#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions for data simulation a specific RSA-model
    identity: One column per unique element in vector
    identity_pos: One column per unique non-zero element
    allpairs:     All n_unique*(n_unique-1)/2 pairwise contrasts
@author: jdiedrichsen
"""

import pyrsa as rsa
import numpy as np
import scipy.stats as ss
import scipy.linalg as sl


def make_design(n_cond, n_part):
    """
    Makes simple fMRI design with n_cond, each measures n_part times

    Args:
        n_cond (int):          Number of conditions
        n_part (int):          Number of partitions
    Returns:
        Tuple (cond_vec, part_vec)
        cond_vec (np.ndarray): n_obs vector with condition
        part_vec (np.ndarray): n_obs vector with partition
    """
    p = np.array(range(0, n_part))
    c = np.array(range(0, n_cond))
    cond_vec = np.kron(np.ones((n_part,)), c) # Condition Vector
    part_vec = np.kron(p,np.ones((n_cond,)))  # Partition vector
    return(cond_vec,part_vec)

def make_dataset(model, theta, cond_vec, n_channel=30, n_sim=1,\
                 signal=1, noise=1, noise_cov=None,\
                 part_vec=None):
    """
    Simulates a fMRI-style data set with a set of partitions

    Args:
        model (rsa.Model):        the model from which to generate data
        theta (numpy.ndarray):    vector of parameters (one dimensional)
        cond_vec (numpy.ndarray): RSA-style model: vector of experimental conditions
                                  Encoding-style model: design matrix (n_obs x n_cond)
        n_channel (int):          Number of channels (default = 30)
        n_sim (int):              Number of simulation with the same signal (default = 1)
        signal (float):           Signal variance (multiplied by predicted G)
        noise (float)             Noise variance (*noise_cov if given)
        noise_cov (numpy.ndarray):n_channel x n_channel covariance matrix of noise (default = identity)
        part_vec (numpy.ndarray): optional partition vector if within-partition covariance is specified
    Returns:
        data (rsa.Dataset):       Dataset with obs_descriptors.
    """

    RDM = model.predict(theta)    # Get the model prediction

    # Make design matrix
    if (cond_vec.ndim == 1):
        Zcond = rsu.indicator.identity(cond_vec)
    elif (cond_vec.ndim == 2):
        Zcond = cond_vec
    else:
        raise(NameError("cond_vec needs to be either condition vector or design matrix"))
    n_obs, n_cond = Zcond.shape

    # If noise_cov given, precalculate the cholinsky decomp
    if (noise_cov is not None):
        if (noise_cov.shape is not (n_channel,n_channel)):
            raise(NameError("noise covariance needs to be n_channel x n_channel array"))
        noise_chol = np.linalg.cholesky(noise_cov)

    # Generate the true patterns with exactly correct second moment matrix
    true_U = np.random.uniform(0, 1, size=(n_cond, n_channel))
    true_U = ss.norm.ppf(true_U)  # We use two-step procedure allow for different distributions later on
    # Make orthonormal row vectors
    E = true_U @ true_U.transpose()
    L = np.linalg.cholesky(E)
    true_U = np.linalg.solve(L, true_U)

    # Now produce data with the known second-moment matrix
    # Use positive eigenvectors only (cholesky does not work with rank-deficient matrices)
    l, V = np.linalg.eig(G)
    l[l<1e-15] = 0
    l = np.sqrt(l)
    chol_G = V.real*l.real.reshape((1, l.size))
    true_U = (chol_G @ true_U) * np.sqrt(n_channel)

    # Generate noise as a matrix normal, independent across partitions
    # If noise covariance structure is given, it is assumed that it's the same
    # across different partitions
    data = np.empty((n_sim, n_obs, n_channel))
    for i in range(0, n_sim):
        epsilon = np.random.uniform(0, 1, size=(n_obs, n_channel))
        epsilon = ss.norm.ppf(epsilon)*np.sqrt(noise)  # Allows alter for providing own cdf for noise distribution
        if (noise_cov is not None):
            epsilon=epsilon @ noise_chol
        data[i,:,:] = Zcond@true_U * np.sqrt(signal) + epsilon
    obs_des = {"cond_vec": cond_vec}
    des     = {"signal": signal,"noise":noise,"model":model.name,"theta": theta}
    dataset = rsa.Dataset(data,obs_descriptors=obs_des,descriptors=des)
    return dataset
