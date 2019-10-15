"""
Created on Sun Oct 13 20:19:10 2019

@author: jdiedrichsen
"""

import pyrsa as rsa
import numpy as np 

class Dataset: 
    """
    Dataset class contains one data set - or multiple data sets with the same structure
    """
    def __init__(self,measurements,descriptors = None ,obs_descriptors = None,channel_descriptors = None): 
        """
        Creator for Dataset class
        
        Args: 
            measurements (numpy.ndarray):   n_obs x n_channel 2d-array, or n_set x n_obs x n_channel 3d-array 
            descriptors (dict):             descriptors with 1 value per Dataset object 
            obs_descriptors (dict):         observation descriptors (all are array-like with shape = (n_obs,...)) 
            channel_descriptors (dict):     channel descriptors (all are array-like with shape = (n_channel,...))
        Returns: 
            dataset object 
        """
        if (measurements.ndim==2):
            self.measurements = measurements
            self.n_set = 1 
            self.n_obs,self.n_channel = self.measurements.shape
        elif (measurements.ndim==3):
            self.measurements = measurements
            self.n_set,self.n_obs,self.n_channel = self.measurements.shape
        self.descriptors = descriptors 
        self.obs_descriptors = obs_descriptors 
        self.channel_descriptors = channel_descriptors 
