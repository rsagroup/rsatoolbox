#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cython
from cython.view cimport array as cvarray
from libc.math cimport log, sqrt
from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free
cimport scipy.linalg.cython_blas as blas
import numpy as np

# np.import_array()


@cython.boundscheck(False)
cpdef double similarity(double [:] vec_i, double [:] vec_j, int method_idx,
                       double [:, :] noise=None,
                       double prior_lambda=1, double prior_weight=0.1):
    """ This is a single similarity computation in cython.
    remember to call everything with continuous numpy arrays.
    In particular, noise must be such an array for Mahalanobis distances!
    """
    cdef double sim = 0
    cdef int n_dim = vec_i.shape[0]
    if method_idx == 1: # method == 'euclidean':
        for i in range(n_dim):
            sim += vec_i[i] * vec_j[i]
    elif method_idx == 2: # method == 'correlation':
        sim = correlation(vec_i, vec_j, n_dim)
    elif method_idx == 3: # method in ['mahalanobis', 'crossnobis']:
        if noise is None:
            sim = similarity(vec_i, vec_j, 1)
        else:
            sim = mahalanobis(vec_i, vec_j, n_dim, noise)
    elif method_idx == 4: # method in ['poisson', 'poisson_cv']:
        sim = poisson_cv(vec_i, vec_j, n_dim, prior_lambda, prior_weight)
    else:
        raise ValueError('dissimilarity method not recognized!')
    return sim


@cython.boundscheck(False)
@cython.cdivision(True)
cdef double poisson_cv(double [:] vec_i, double [:] vec_j, int n_dim,
                       double prior_lambda, double prior_weight):
    cdef:
        double prior_lambda_l = prior_lambda * prior_weight
        double prior_weight_l = 1 + prior_weight
        double vi, vj
        double sim = 0
    for i in range(n_dim):
        vi = (vec_i[i] + prior_lambda_l) / prior_weight_l
        vj = (vec_i[i] + prior_lambda_l) / prior_weight_l
        sim += (vj - vi) * (log(vi) - log(vj))
    sim = sim / 2.0
    return sim


@cython.boundscheck(False)
cdef double mahalanobis(double [:] vec_i, double [:] vec_j, int n_dim,
                        double [:, :] noise):
    cdef:
        double *vec2
        int zero = 0
        int one = 1
        double onef = 1.0
        double zerof = 0.0
        char trans = 'n'
        double sim = 0.0
    #vec2 = cvarray(shape=n_dim, itemsize=sizeof(double), format="d")
    vec2 = <double*> PyMem_Malloc(n_dim * sizeof(double))
    #blas.dgemv(&1.0, &noise[0], &vec_j[0], &0.0, &vec2[0], &zero, &one, &zero, &one, &zero, &one)
    #blas.dgemv(&onef, &noise[0, 0], &vec_j[0], &zerof, &vec2[0], &zerof, &one, &zero, &one, &zero, &one)
    blas.dgemv(&trans, &n_dim, &n_dim, &onef, &noise[0, 0], &n_dim, &vec_j[0], &one, &zerof, vec2, &one)
    for i in range(n_dim):
        sim += vec_i[i] * vec2[i]
    PyMem_Free(vec2)
    return sim


@cython.boundscheck(False)
@cython.cdivision(True)
cdef double correlation(double [:] vec_i, double [:] vec_j, int n_dim):
    cdef:
        double si = 0.0
        double sj = 0.0
        double si2 = 0.0
        double sj2 = 0.0
        double sij = 0.0
        double sim
    for i in range(n_dim):
        si += vec_i[i]
        sj += vec_j[i]
        si2 += vec_i[i] * vec_i[i]
        sj2 += vec_j[i] * vec_j[i]
        sij += vec_i[i] * vec_j[i]
    if si2 > 0 and sj2 > 0:
        # sim = (np.sum(vec_i * vec_j) / np.sqrt(norm_i) / np.sqrt(norm_j))
        sim = sij - si * sj
        sim /= sqrt(si2 - si * si)
        sim /= sqrt(sj2 - sj * sj)
    else:
        sim = 1
    sim = sim * n_dim
    return sim