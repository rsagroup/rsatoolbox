#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions for data simulation a specific RSA-model

    make_design: creates design and condition vectors for fMRI design

    make_dataset: creates a data set based on an RDM model

@author: jdiedrichsen
"""
import numpy as np
import scipy.stats as ss
import scipy.linalg as sl
from scipy.spatial.distance import squareform
import rsatoolbox


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
    cond_vec = np.kron(np.ones((n_part,)), c)   # Condition Vector
    part_vec = np.kron(p, np.ones((n_cond,)))    # Partition vector
    return (cond_vec, part_vec)


def make_dataset(model, theta, cond_vec, n_channel=30, n_sim=1,
                 signal=1, noise=1, signal_cov_channel=None,
                 noise_cov_channel=None, noise_cov_trial=None,
                 use_exact_signal=False, use_same_signal=False):
    """
    Simulates a fMRI-style data set

    Args:
        model (rsatoolbox.Model):        the model from which to generate data
        theta (numpy.ndarray):    vector of parameters (one dimensional)
        cond_vec (numpy.ndarray): RSA-style model:
                                      vector of experimental conditions
                                  Encoding-style:
                                      design matrix (n_obs x n_cond)
        n_channel (int):          Number of channels (default = 30)
        n_sim (int):              Number of simulation with the same signal
                                      (default = 1)
        signal (float):            Signal variance (multiplied by predicted G)
        signal_cov_channel(numpy.ndarray):
            Covariance matrix of signal across channels
        noise (float):
            Noise variance
        noise_cov_channel(numpy.ndarray):
            Covariance matrix of noise (default = identity)
        noise_cov_trial(numpy.ndarray):
            Covariance matrix of noise across trials
        use_exact_signal (bool):  Makes the signal so that G is exactly as
                                  specified (default: False)
        use_same_signal (bool):   Uses the same signal for all simulation
                                  (default: False)
    Returns:
        data (list):              List of rsatoolbox.Dataset with obs_descriptors
    """

    # Get the model prediction and build second moment matrix
    # Note that this step assumes that RDM uses squared Euclidean distances
    RDM = model.predict(theta)
    D = squareform(RDM)
    H = rsatoolbox.util.matrix.centering(D.shape[0])
    G = -0.5 * (H @ D @ H)

    # Make design matrix
    if cond_vec.ndim == 1:
        Zcond = rsatoolbox.util.matrix.indicator(cond_vec)
    elif cond_vec.ndim == 2:
        Zcond = cond_vec
    else:
        raise ValueError("cond_vec needs to be either vector or design matrix")
    n_obs, _ = Zcond.shape

    # If signal_cov_channel is given, precalculate the cholesky decomp
    if signal_cov_channel is None:
        signal_chol_channel = None
    else:
        if signal_cov_channel.shape != (n_channel, n_channel):
            raise ValueError("Signal covariance for channels needs to be \
                              n_channel x n_channel array")
        signal_chol_channel = np.linalg.cholesky(signal_cov_channel)

    # If noise_cov_channel is given, precalculate the cholinsky decomp
    if noise_cov_channel is None:
        noise_chol_channel = None
    else:
        if noise_cov_channel.shape != (n_channel, n_channel):
            raise ValueError("noise covariance for channels needs to be \
                              n_channel x n_channel array")
        noise_chol_channel = np.linalg.cholesky(noise_cov_channel)

    # If noise_cov_trial is given, precalculate the cholinsky decomp
    if noise_cov_trial is None:
        noise_chol_trial = None
    else:
        if noise_cov_trial.shape != (n_channel, n_channel):
            raise ValueError("noise covariance for trials needs to be \
                              n_obs x n_obs array")
        noise_chol_trial = np.linalg.cholesky(noise_cov_trial)

    # Generate the signal - here same for all simulations
    if use_same_signal:
        true_U = make_signal(G, n_channel, use_exact_signal,
                             signal_chol_channel)

    # Generate noise as a matrix normal, independent across partitions
    # If noise covariance structure is given, it is assumed that it's the same
    # across different partitions
    obs_des = {"cond_vec": cond_vec}
    des = {"signal": signal, "noise": noise,
           "model": model.name, "theta": theta}
    dataset_list = []
    for _ in range(0, n_sim):
        # If necessary - make a new signal
        if not use_same_signal:
            true_U = make_signal(G, n_channel, use_exact_signal,
                                 signal_chol_channel)
        # Make noise with normal distribution
        # - allows later plugin of other dists
        epsilon = np.random.uniform(0, 1, size=(n_obs, n_channel))
        epsilon = ss.norm.ppf(epsilon) * np.sqrt(noise)
        # Now add spatial and temporal covariance structure as required
        if noise_chol_channel is not None:
            epsilon = epsilon @ noise_chol_channel
        if noise_chol_trial is not None:
            epsilon = noise_chol_trial @ epsilon
        # Assemble the data set
        data = Zcond @ true_U * np.sqrt(signal) + epsilon
        dataset = rsatoolbox.data.Dataset(data,
                                     obs_descriptors=obs_des,
                                     descriptors=des)
        dataset_list.append(dataset)
    return dataset_list


def make_signal(G, n_channel, make_exact=False, chol_channel=None):
    """
    Generates signal exactly with a specified second-moment matrix (G)

    To avoid errors: If the number of channels is smaller than the
    number of patterns we generate a representation with the minimal
    number of dimnensions and then delete dimensions to yield the desired
    number of dimensions.

    Args:
        G(np.array)        : desired second moment matrix (ncond x ncond)
        n_channel (int)    : Number of channels
        make_exact (bool)  : enforce exact signal distances
                             (default: False)
        chol_channel: Cholensky decomposition of the signal covariance matrix
                             (default: None - makes signal i.i.d.)
    Returns:
        np.array (n_cond x n_channel): random signal

    """
    # Generate the true patterns with exactly correct second moment matrix
    n_cond = G.shape[0]
    if n_cond > n_channel:
        n_channel_final = n_channel
        n_channel = n_cond
    else:
        n_channel_final = None
    # We use two-step procedure allow for different distributions later on
    true_U = np.random.uniform(0, 1, size=(n_cond, n_channel))
    true_U = ss.norm.ppf(true_U)
    true_U = true_U - np.mean(true_U, axis=1, keepdims=True)
    # Make orthonormal row vectors
    if make_exact:
        E = true_U @ true_U.transpose()
        L_E, D_E, _ = sl.ldl(E)
        D_E[D_E < 1e-15] = 1e-15  # we need an invertible solution!
        D_E = np.sqrt(D_E)
        E_chol = L_E @ D_E
        true_U = np.linalg.solve(E_chol, true_U) * np.sqrt(n_channel)
    # Impose spatial covariance matrix
    if chol_channel is not None:
        true_U = true_U @ chol_channel
    # Now produce data with the known second-moment matrix
    # Use positive eigenvectors only
    # (cholesky does not work with rank-deficient matrices)
    L, D, _ = sl.ldl(G)
    D[D < 1e-15] = 0
    D = np.sqrt(D)
    chol_G = L @ D
    true_U = (chol_G @ true_U)
    if n_channel_final:
        true_U = true_U[:, :n_channel_final]
    return true_U
