#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 10:50:15 2019

@author: heiko
"""
import stats
import nn_simulations as dnn


#Profiling
def main():
    #stim = stats.get_stimuli_92()
    #RDM_true = dnn.get_true_RDM(None,3,stim[0:4])
    stats.pipeline_pooling_dnn(Nsubj=5,Nsamp=2,Nvox=250,Nstimuli=50)
    
if __name__ == '__main__':
    main()