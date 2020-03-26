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
        subset = data.subset_obs(by='conds',value=[2,3])
        self.assertEqual(subset.n_obs,4)
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
        subset = data.subset_channel(by='rois',value=['IT','V4'])
        self.assertEqual(subset.n_obs,10)
        self.assertEqual(subset.n_channel,3)
        self.assertEqual(subset.channel_descriptors['rois'][0],'IT')
        self.assertEqual(subset.channel_descriptors['rois'][-1],'V4')


class TestDataComputations(unittest.TestCase):
    def setUp(self):
        measurements = np.random.rand(10,5)
        des = {'session':0,'subj':0}
        obs_des = {'conds':np.array([0,0,1,1,2,2,2,3,4,5])}
        chn_des = {'rois':np.array(['V1','V1','IT','IT','V4'])}
        self.test_data = rsd.Dataset(measurements=measurements,
                           descriptors=des,
                           obs_descriptors=obs_des,
                           channel_descriptors=chn_des
                           )
        
    def test_average(self):
        avg = rsd.average_dataset(self.test_data)
        self.assertEqual(avg.shape,(5,))
        
    def test_average_by(self):
        avg,descriptor = rsd.average_dataset_by(self.test_data,'conds')
        self.assertEqual(avg.shape, (6,5))
        self.assertEqual(len(descriptor),6)
        self.assertEqual(descriptor[-1],5)
        assert(np.all(self.test_data.measurements[-1]==avg[-1]))
        

class testSave(unittest.TestCase):
    def test_dict_conversion(self):
        measurements = np.zeros((10,5))
        des = {'session':0,'subj':0}
        obs_des = {'conds':np.array([0,0,1,1,2,2,2,3,4,5])}
        chn_des = {'rois':np.array(['V1','V1','IT','IT','V4'])}
        data = rsd.Dataset(measurements=measurements,
                           descriptors=des,
                           obs_descriptors=obs_des,
                           channel_descriptors=chn_des
                           )
        data_dict = data.to_dict()
        data_loaded = rsd.dataset_from_dict(data_dict)
        assert type(data_loaded) == type(data)
        assert data_loaded.n_channel == data.n_channel
        assert np.all(data_loaded.obs_descriptors['conds'] == obs_des['conds'])
        assert np.all(data_loaded.channel_descriptors['rois'] == chn_des['rois'])
        assert data_loaded.descriptors['subj'] == 0
        
        
    def test_save_load(self):
        import io
        f = io.BytesIO() # Essentially a Mock file
        measurements = np.zeros((10,5))
        des = {'session':0,'subj':0}
        obs_des = {'conds':np.array([0,0,1,1,2,2,2,3,4,5])}
        chn_des = {'rois':np.array(['V1','V1','IT','IT','V4'])}
        data = rsd.Dataset(measurements=measurements,
                           descriptors=des,
                           obs_descriptors=obs_des,
                           channel_descriptors=chn_des
                           )
        data.save(f, file_type='hdf5')
        data_loaded = rsd.load_dataset(f, file_type='hdf5')
        assert data_loaded.n_channel == data.n_channel
        assert np.all(data_loaded.obs_descriptors['conds'] == obs_des['conds'])
        assert np.all(data_loaded.channel_descriptors['rois'] == chn_des['rois'])
        assert data_loaded.descriptors['subj'] == 0
        

if __name__ == '__main__':
    unittest.main()  
