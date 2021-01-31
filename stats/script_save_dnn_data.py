#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 16:31:12 2019

@author: heiko
"""

import stats
import numpy as np
import tqdm

output_folder = '/media/heiko/Disk/rsa_simulations/test'

Nsim = 100
# 180 simulations x12 layers x 6 numbers of stimuli
Nsubj = np.array((5, 10, 20, 40, 80))
Nvox = np.array((50, 100, 250, 500))
Nrepeats = np.array((2, 8, 32))
noise_stds = np.array((0.5, 2, 8))
Nstimuli = np.array((6, 12, 24, 48, 96))


for iNsubj in tqdm.tqdm(Nsubj):
    for iNvox in tqdm.tqdm(Nvox):
        for iStd in tqdm.tqdm(noise_stds):
            for iLayer in tqdm.trange(12):
                stats.save_simulated_data_dnn(simulation_folder=output_folder,
                                              layer=iLayer + 1,
                                              Nsubj=iNsubj, Nvoxel=iNvox,
                                              Nsim=Nsim, sigma_noise=iStd,
                                              duration=1)
            for iLayer in tqdm.trange(12):
                for iNstimuli in tqdm.tqdm(Nstimuli):
                    stats.analyse_saved_dnn(simulation_folder=output_folder,
                                            layer=iLayer+1, Nsubj=iNsubj,
                                            Nvoxel=iNvox, Nsim=Nsim,
                                            Nstimuli=iNstimuli,
                                            sigma_noise=iStd,
                                            NLayer=12, duration=1,
                                            RDM_comparison='cosine')
                    stats.analyse_saved_dnn(simulation_folder=output_folder,
                                            layer=iLayer+1, Nsubj=iNsubj,
                                            Nvoxel=iNvox, Nsim=Nsim,
                                            Nstimuli=iNstimuli,
                                            sigma_noise=iStd,
                                            NLayer=12, duration=1,
                                            RDM_comparison='euclid')
                    stats.analyse_saved_dnn(simulation_folder=output_folder,
                                            layer=iLayer+1, Nsubj=iNsubj,
                                            Nvoxel=iNvox, Nsim=Nsim,
                                            Nstimuli=iNstimuli,
                                            sigma_noise=iStd,
                                            NLayer=12, duration=1,
                                            RDM_comparison='spearman')
                    stats.analyse_saved_dnn(simulation_folder=output_folder,
                                            layer=iLayer+1, Nsubj=iNsubj,
                                            Nvoxel=iNvox, Nsim=Nsim,
                                            Nstimuli=iNstimuli,
                                            sigma_noise=iStd,
                                            NLayer=12, duration=1,
                                            RDM_comparison='corr')
                    stats.analyse_saved_dnn(simulation_folder=output_folder,
                                            layer=iLayer+1, Nsubj=iNsubj,
                                            Nvoxel=iNvox, Nsim=Nsim,
                                            Nstimuli=iNstimuli,
                                            sigma_noise=iStd,
                                            NLayer=12, duration=1,
                                            RDM_comparison='kendall-tau')
                    stats.analyse_saved_dnn(simulation_folder=output_folder,
                                            layer=iLayer+1, Nsubj=iNsubj,
                                            Nvoxel=iNvox, Nsim=Nsim,
                                            Nstimuli=iNstimuli,
                                            sigma_noise=iStd,
                                            NLayer=12, duration=1,
                                            RDM_comparison='tau-a')
