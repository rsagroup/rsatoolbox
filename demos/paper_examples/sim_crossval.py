import rsatoolbox as rsa
import numpy as np
import PcmPy as pcm
import scipy.spatial.distance as sd
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt

# Simulate data

def crossval_sim(n_part=2,n_cond=4,n_sim=10,n_channel=200,sigma='iid'):
    """ Simulate data from a model with 4 conditions
    Use different noise covriances across trials for each
    Use square eucledian or cross-validated square Eucldian distancen
    Low numbers of partitions to emphasize increase in variance.
    """

    cond_vec,part_vec = rsa.simulation.make_design(n_cond,n_part)
    true_dist = np.array([2,1,0,3,2,1])
    dist_type = np.array([1,2,3,4,5,6])
    D = sd.squareform(true_dist)
    H = pcm.matrix.centering(n_cond)
    G = -0.5 * H @ D @ H
    M  = pcm.model.FixedModel('fixed',G)
    if (sigma=='iid'):
        Sigma = np.kron(np.eye(n_part),np.eye(n_cond))
    elif (sigma=='neigh'):
        A = [[1,0.8,0,0],[0.8,1,0,0.5],[0,0,1,0],[0,0.5,0,1]]
        Sigma = np.kron(np.eye(n_part),A)
    data = pcm.sim.make_dataset(M,[],cond_vec,
                         n_sim=n_sim,
                         noise=4,
                         n_channel=n_channel,
                         noise_cov_trial=Sigma)
    Z = pcm.matrix.indicator(cond_vec)

    D_simp = np.zeros((n_sim,n_cond*(n_cond-1)//2))
    D_cross = np.zeros((n_sim,n_cond*(n_cond-1)//2))
    for i in range(n_sim):
        mean_act = np.linalg.pinv(Z) @ data[i].measurements
        D_simp[i] = sd.pdist(mean_act)**2/n_channel
        G_cross,_ = pcm.est_G_crossval(data[i].measurements,cond_vec,part_vec)
        D_cross[i,:] = sd.squareform(pcm.G_to_dist(G_cross))


    # model, theta, cond_vec, n_channel=30, n_sim=1,
    #              signal=1, noise=1, signal_cov_channel=None,
    #              noise_cov_channel=None, noise_cov_trial=None,
    #              use_exact_signal=False, use_same_signal=False)
    T=pd.DataFrame({'Simp':D_simp.flatten(),
                    'Cross':D_cross.flatten(),
                    'True':np.tile(true_dist,n_sim),
                    'dist_type':np.tile(dist_type,n_sim)})
    return(T)

def plot_panel(T):
    sb.violinplot(data=T,x='True',y='Simp')
    sb.violinplot(data=T,x='True',y='Cross')
    sb.despine()
    plt.plot([0,3],[0,3],'k--')
    plt.xlabel('True distance')
    plt.ylabel('Estimated distance')
    ax=plt.gca()
    ax.set_ylim([-1,10])


if __name__=="__main__":
    T1 = crossval_sim(sigma='iid',n_sim=100)
    T2 = crossval_sim(sigma='neigh',n_sim=100)
    plt.figure()
    plt.subplot(1,2,1)
    plot_panel(T1)
    plt.subplot(1,2,2)
    plot_panel(T2)

    pass