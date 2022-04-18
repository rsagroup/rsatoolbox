#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Composite test for modelfamily class
@author: jdiedrichsen
"""

#pylint: disable=import-outside-toplevel, no-self-use
import unittest
import numpy as np
from scipy.spatial.distance import squareform
import rsatoolbox as rsa
import rsatoolbox.model as model
import matplotlib.pyplot as plt
from scipy import linalg


def two_by_three_design(orthogonalize=False):
    A = np.array([[1.0,0,0],[1,0,0],[0,1,0],[0,1,0],[0,0,1],[0,0,1]])
    B = np.array([[1.0,0],[0,1],[1,0],[0,1],[1,0],[0,1]])
    I = np.eye(6)
    if orthogonalize:
        A = A - A.mean(axis=0)
        B = B - B.mean(axis=0)
        I = I - I.mean(axis=0)
        X= np.c_[A,B]
        I = I-X @ linalg.pinv(X) @ I
    rdms = np.zeros((3,6,6))
    C =  rsa.util.matrix.pairwise_contrast(np.arange(6))
    rdms[0]=squareform(np.diag(C@A@A.T@C.T))
    rdms[1]=squareform(np.diag(C@B@B.T@C.T))
    rdms[2]=squareform(np.diag(C@I@I.T@C.T))

    M = model.ModelWeighted('A+B+I',rdms)
    MF= model.ModelFamily(rdms,comp_names=['A','B','I'])
    return M,MF

def component_inference(D,MF):
    data_rdms = rsa.rdm.calc_rdm(D,method='crossnobis',
                            descriptor='cond_vec',
                            cv_descriptor='part_vec')
    Res=rsa.inference.eval_fixed(MF.models,data_rdms,method='cosine')

    # pcm.vis.model_plot(T.likelihood-MF.num_comp_per_m)
    mposterior = MF.model_posterior(Res,method=None,format='ndarray')
    cposterior = MF.component_posterior(Res,method=None,format='DataFrame')
    c_bf = MF.component_bayesfactor(Res,method=None,format='DataFrame')

    fig=plt.figure(figsize=(6,6))
    plt.subplot(2,2,1)
    rsa.vis.family_graph(MF,node_facecolor=mposterior[0])
    ax=plt.subplot(2,2,2)
    rsa.vis.family_graph(MF,node_size=mposterior[0])
    ax=plt.subplot(2,2,3)
    rsa.vis.component_barplot(cposterior,type='posterior')
    ax=plt.subplot(2,2,4)
    rsa.vis.component_barplot(c_bf,type='bf')
    pass

def sim_two_by_three(theta):
    """Simulates a simple 2x3 factorial design
    Using non-orthogonalized contrasts
    """
    M,MF1 = two_by_three_design(orthogonalize=False)
    M,MF2 = two_by_three_design(orthogonalize=True)
    [cond_vec,part_vec]=rsa.simulation.make_design(6,8)
    D = rsa.simulation.make_dataset(M,
                            theta=np.array([0.7,0,0]),
                            cond_vec = cond_vec,
                            signal=0.1,
                            n_sim = 20,
                            n_channel=20,
                            part_vec = part_vec)
    component_inference(D,MF1)
    component_inference(D,MF2)
    pass

def random_design(N=10,Q=5,num_feat=2,seed=1):
    rng = np.random.default_rng(seed)
    C =  rsa.util.matrix.pairwise_contrast(np.arange(N))
    rdms = np.zeros((Q,N,N))
    for q in range(Q):
        X= rng.normal(0,1,(N,num_feat))
        rdms[q]=squareform(np.diag(C@X@X.T@C.T))
    M = model.ModelWeighted('full',rdms)
    MF= model.ModelFamily(rdms)
    return M,MF

def random_example(theta,N=10):
    Q = theta.shape[0]
    M,MF = random_design(N=N,Q=Q)
    rsa.vis.show_rdm(M.rdm_obj)

    [cond_vec,part_vec]=rsa.simulation.make_design(N,8)
    D = rsa.simulation.make_dataset(M,theta,
                            signal=1.0,
                            n_sim = 20,
                            n_channel=20,
                            part_vec=part_vec,
                            cond_vec=cond_vec)
    component_inference(D,MF)
    pass



if __name__ == '__main__':
    # sim_two_by_three(np.array([0,0.0,20]))
    random_example(np.array([0,0,0,0,1]))

