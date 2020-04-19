#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 11:28:16 2020

@author: heiko
"""


#import os
#import scipy.io
import numpy as np
import pyrsa
from pyrsa.util.matrix import pairwise_contrast_sparse
from pyrsa.rdm.calc import _calc_pairwise_differences
import time

n_subjects = 10
subject_pattern_noise_std = 1


#from pyrsa.util.matrix import pairwise_contrast_sparse
#test = pairwise_contrast_sparse(np.arange(1000))

#from pyrsa.util.matrix import pairwise_contrast
#pairwise_contrast(np.arange(1000))

measurements = np.random.rand(1000,100)
#dat = pyrsa.data.Dataset(measurements)
#rdm = pyrsa.rdm.calc_rdm(dat)

# data_matlab2 = scipy.io.loadmat(os.path.join('92imageData',
#     '92_brainRDMs.mat'))

# rdms_sess1 = np.array([data_matlab2['RDMs'][0][i][0][0] for i in range(4)])
# rdms_sess2 = np.array([data_matlab2['RDMs'][0][i][1][0] for i in range(4)])
# rdms_array = np.concatenate((rdms_sess1,rdms_sess2),0)
# rdms_human = pyrsa.rdm.RDMs(rdms_array, rdm_descriptors = {
#     'session':np.array([1,1,1,1,2,2,2,2]),
#     'subject':np.array([1,2,3,4,1,2,3,4])})

# # TODO: plot these rdms!

# # load reconstructed patterns for simulating models
# data_matlab3 = scipy.io.loadmat(os.path.join('92imageData',
#     'simTruePatterns.mat'))
# sim_true_patterns = data_matlab3['simTruePatterns']
# sim_true_patterns2 = data_matlab3['simTruePatterns2']
# n_cond, n_dim = sim_true_patterns.shape

# # simulate multiple subjects' noisy RDMs

# data_list = []
# for i_subject in range(n_subjects):
#     patterns_subject = sim_true_patterns2 \
#         + subject_pattern_noise_std * np.random.randn(n_cond, n_dim)
#     dataset = pyrsa.data.Dataset(patterns_subject)
#     data_list.append(dataset)
# subject_rdms = pyrsa.rdm.calc_rdm(data_list)


t0 = time.time()
c_matrix = pairwise_contrast_sparse(np.arange(measurements.shape[0]))
diff = c_matrix @ measurements
t1 = time.time()
diff2 = np.zeros_like(diff)
k = 0
for i in range(measurements.shape[0]):
    for j in range(i+1, measurements.shape[0]):
        diff2[k] = measurements[i] - measurements[j]
        k += 1
t2 = time.time()
diff3 = _calc_pairwise_differences(measurements)
t3 = time.time()
