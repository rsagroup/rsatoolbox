#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_data
Test for Dataset class 
@author: baihan, jdiedrichsen
"""

import unittest 
import pyrsa.data as rsd
import numpy as np 


class TestData(unittest.TestCase): 
    
    def test_dataset_simple_init(self):
        measurements = np.zeros((10,5))
        data = rsd.Dataset(measurements)
        self.assertEqual(data.n_obs,10)
        self.assertEqual(data.n_channel,5)

    def test_dataset_full_init(self):
        measurements = np.zeros((10,5))
        des = {'session':0,'subj':0}
        obs_des = {'conds':np.array(['cond_'+str(x) for x in np.arange(10)])}
        chn_des = {'rois':np.array(['roi_'+str(x) for x in np.arange(5)])}
        data = rsd.Dataset(measurements=measurements,
                           descriptors=des,
                           obs_descriptors=obs_des,
                           channel_descriptors=chn_des
                           )
        self.assertEqual(data.n_obs,10)
        self.assertEqual(data.n_channel,5)
        self.assertEqual(data.descriptors,des)
        self.assertEqual(data.obs_descriptors,obs_des)
        self.assertEqual(data.channel_descriptors,chn_des)

    def test_dataset_split_obs(self):
        measurements = np.zeros((10,5))
        des = {'session':0,'subj':0}
        obs_des = {'conds':np.array([0,0,1,1,2,2,2,3,4,5])}
        chn_des = {'rois':np.array(['V1','V1','IT','IT','V4'])}
        data = rsd.Dataset(measurements=measurements,
                           descriptors=des,
                           obs_descriptors=obs_des,
                           channel_descriptors=chn_des
                           )
        splited_list = data.split_obs('conds')
        self.assertEqual(len(splited_list),6)
        self.assertEqual(splited_list[0].n_obs,2)
        self.assertEqual(splited_list[2].n_obs,3)
        self.assertEqual(splited_list[0].n_channel,5)
        self.assertEqual(splited_list[2].n_channel,5)
        self.assertEqual(splited_list[2].obs_descriptors['conds'][0],2)

    def test_dataset_split_channel(self):
        measurements = np.zeros((10,5))
        des = {'session':0,'subj':0}
        obs_des = {'conds':np.array([0,0,1,1,2,2,2,3,4,5])}
        chn_des = {'rois':np.array(['V1','V1','IT','IT','V4'])}
        data = rsd.Dataset(measurements=measurements,
                           descriptors=des,
                           obs_descriptors=obs_des,
                           channel_descriptors=chn_des
                           )
        splited_list = data.split_channel('rois')
        self.assertEqual(len(splited_list),3)
        self.assertEqual(splited_list[0].n_obs,10)
        self.assertEqual(splited_list[2].n_obs,10)
        self.assertEqual(splited_list[0].n_channel,2)
        self.assertEqual(splited_list[2].n_channel,1)
        self.assertEqual(splited_list[1].channel_descriptors['rois'][0],'IT')

    def test_dataset_subset_obs(self):
        measurements = np.zeros((10,5))
        des = {'session':0,'subj':0}
        obs_des = {'conds':np.array([0,0,1,1,2,2,2,3,4,5])}
        chn_des = {'rois':np.array(['V1','V1','IT','IT','V4'])}
        data = rsd.Dataset(measurements=measurements,
                           descriptors=des,
                           obs_descriptors=obs_des,
                           channel_descriptors=chn_des
                           )
        subset = data.subset_obs(by='conds',value=2)
        self.assertEqual(subset.n_obs,3)
        self.assertEqual(subset.n_channel,5)
        self.assertEqual(subset.obs_descriptors['conds'][0],2)

    def test_dataset_subset_channel(self):
        measurements = np.zeros((10,5))
        des = {'session':0,'subj':0}
        obs_des = {'conds':np.array([0,0,1,1,2,2,2,3,4,5])}
        chn_des = {'rois':np.array(['V1','V1','IT','IT','V4'])}
        data = rsd.Dataset(measurements=measurements,
                           descriptors=des,
                           obs_descriptors=obs_des,
                           channel_descriptors=chn_des
                           )
        subset = data.subset_channel(by='rois',value='IT')
        self.assertEqual(subset.n_obs,10)
        self.assertEqual(subset.n_channel,2)
        self.assertEqual(subset.channel_descriptors['rois'][0],'IT')

if __name__ == '__main__':
    unittest.main()  