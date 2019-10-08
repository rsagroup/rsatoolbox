#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Definition of RSA Model class and subclasses 
@author: jdiedrichsen, heiko
"""

import numpy as np


"""
Abstract model class. 
Defines members that every class needs to have, but does not implement any
interesting behavior. Inherit from this class to define specific model types
"""
class Model:
    # Model Constructor
    def __init__(self, name):
        self.name = name
        self.n_param = 0
        
    # The predict function should return a rdm vector
    def predict(self,theta):
        raise(NameError("This function needs to be implemented in the derived class"))
        
    
"""
Fixed model
This is a parameter-free model that simply predicts a fixed RDM
It takes a unidimension numpy-vector as an input to define the RDM
"""
class ModelFixed(Model):
    # Model Constructor
    def __init__(self, name, rdm):
        Model.__init__(self, name)
        if (rdm.ndim == 1):  # User passed a vector
            self.n_cond = (1+np.sqrt(1+8*rdm.size))/2
            if (self.n_cond%1!=0):
                raise (NameError("RDM vector needs to have size of ncond*(ncond-1)/2"))
            self.rdm = rdm   # Add check to make sure it's
            self.n_param = 0

    # prediction returns the 
    def predict(self,theta):
        return self.rdm

