#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 15:39:20 2019

@author: heiko
"""

import numpy as np

class Model:
    def __init__(self,name):
        self.default_fitter = None
        self.name = name
        self.n_param = 0
    def predict(self):
        return np.ones((5,5))-np.eye(5)
    def fit(self,data):
        if self.default_fitter is None:
            return None
        else:
            return self.default_fitter(self,data)
        
