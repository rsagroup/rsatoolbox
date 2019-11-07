#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions for data simulation a specific RSA-model
    make_design: creates design and condition vectors for fMRI design
    make_dataset: creates a data set based on an RDM model
@author: jdiedrichsen
"""

import pyrsa as rsa
import numpy as np
import scipy.stats as ss
import scipy.linalg as sl
from scipy.spatial.distance import squareform


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
                 signal=1, noise=1, noise_cov=None, part_vec=None):
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
        data (list):              List of rsa.Dataset with obs_descriptors
    """

    # Get the model prediction and build second moment matrix
    # Note that this step assumes that RDM uses squared Euclidean distances
    RDM = model.predict(theta)
    D = squareform(RDM)
    H = rsa.util.matrix.centering(D.shape[0])
    G = -0.5 * (H @ D @ H)

    # Make design matrix
    if (cond_vec.ndim == 1):
        Zcond = rsa.util.matrix.indicator(cond_vec)
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

    # Generate the signal - here same for all simulations
    true_U = make_exact_signal(G,n_channel)

    # Generate noise as a matrix normal, independent across partitions
    # If noise covariance structure is given, it is assumed that it's the same
    # across different partitions
    obs_des = {"cond_vec": cond_vec}
    des     = {"signal": signal,"noise":noise,"model":model.name,"theta": theta}
    dataset_list = []
    for i in range(0, n_sim):
        epsilon = np.random.uniform(0, 1, size=(n_obs, n_channel))
        epsilon = ss.norm.ppf(epsilon)*np.sqrt(noise)  # Allows alter for providing own cdf for noise distribution
        if (noise_cov is not None):
            epsilon=epsilon @ noise_chol
        data = Zcond @ true_U * np.sqrt(signal) + epsilon
        dataset = rsa.data.Dataset(data,obs_descriptors=obs_des,descriptors=des)
        dataset_list.append(dataset)
    return dataset_list


def make_exact_signal(G,n_channel):
    """
    Generates signal exactly with a specified second-moment matrix (G)
    Args:
        G(np.array): desired second moment matrix (ncond x ncond)
        n_channel (int) : Number of channels 
    Returns:
        np.array (n_cond x n_channel): random signal 
    """
    # Generate the true patterns with exactly correct second moment matrix
    n_cond = G.shape[0]
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
    return true_U