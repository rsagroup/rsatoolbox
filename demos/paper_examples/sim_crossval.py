# import rsatoolbox as rsa
import numpy as np
import PcmPy as pcm
import scipy.spatial.distance as sd

# Simulate data

def crossval_sim():
    # Simulate data from a model with 4 conditions
    # Use different noise covriances across trials for each
    # Use square eucledian or cross-validated square Eucldian distancen 
    # Low numbers of partitions to emphasize increase in variance. 
    n_part = 3 
    n_cond = 4 
    n_sim = 10
    cond_vec,part_vec = pcm.sim.make_design(n_cond,n_part)
    true_dist = np.array([0,1,2.5,0,2,1])
    D = sd.squareform(true_dist)
    H = pcm.matrix.centering(n_cond)
    G = -0.5 * H @ D @ H
    M  = pcm.model.FixedModel('fixed',G)
    Sigma = np.kron(np.eye(n_part),np.eye(n_cond))
    data = pcm.sim.make_dataset(M,[],cond_vec,
                         n_sim=n_sim,
                         noise=1,
                         noise_cov_trial=Sigma)
    Z = pcm.matrix.indicator(cond_vec)

    D_simp = np.zeros((n_sim,n_cond*(n_cond-1)//2))
    D_cross = np.zeros((n_sim,n_cond*(n_cond-1)//2))
    for i in range(n_sim):
        mean_act = np.linalg.pinv(Z) @ data[i].measurements
        D_simp[i] = sd.pdist(mean_act)
        G_cross,_ = pcm.est_G_crossval(data[i].measurements,cond_vec,part_vec)
        D_cross[i,:] = sd.squareform(pcm.G_to_dist(G_cross))

    G_est = pcm.est_G_crossval(data,cond_vec,part_vec)

    # model, theta, cond_vec, n_channel=30, n_sim=1,
    #              signal=1, noise=1, signal_cov_channel=None,
    #              noise_cov_channel=None, noise_cov_trial=None,
    #              use_exact_signal=False, use_same_signal=False)
    pass 



if __name__=="__main__":
    crossval_sim()
    pass