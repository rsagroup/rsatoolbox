#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 15:09:36 2019

@author: heiko
"""

import stats

stats.pipeline_pooling_dnn(Nsubj=10,Nsamp=100,Nvox=500,Nstimuli=40,layer=1)
stats.pipeline_pooling_dnn(Nsubj=10,Nsamp=100,Nvox=200,Nstimuli=40,layer=1)
stats.pipeline_pooling_dnn(Nsubj=10,Nsamp=100,Nvox=75,Nstimuli=40,layer=1)
stats.pipeline_pooling_dnn(Nsubj=10,Nsamp=100,Nvox=25,Nstimuli=40,layer=1)
stats.pipeline_pooling_dnn(Nsubj=10,Nsamp=100,Nvox=10,Nstimuli=40,layer=1)