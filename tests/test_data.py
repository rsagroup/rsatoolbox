#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_data
Test for Dataset class
@author: baihan, jdiedrichsen, adkipnis
"""

import unittest
import pyrsa.data as rsd
import numpy as np


class TestData(unittest.TestCase):

    def test_dataset_simple_init(self):
        measurements = np.zeros((10, 5))
        data = rsd.Dataset(measurements)
        self.assertEqual(data.n_obs, 10)
        self.assertEqual(data.n_channel, 5)

    def test_dataset_full_init(self):
        measurements = np.zeros((10, 5))
        des = {'session': 0, 'subj': 0}
        obs_des = {'conds': np.array(['cond_' + str(x)
                                      for x in np.arange(10)])}
        chn_des = {'rois': np.array(['roi_' + str(x) for x in np.arange(5)])}
        data = rsd.Dataset(measurements=measurements,
                           descriptors=des,
                           obs_descriptors=obs_des,
                           channel_descriptors=chn_des
                           )
        self.assertEqual(data.n_obs, 10)
        self.assertEqual(data.n_channel, 5)
        self.assertEqual(data.descriptors, des)
        self.assertEqual(data.obs_descriptors, obs_des)
        self.assertEqual(data.channel_descriptors, chn_des)

    def test_dataset_split_obs(self):
        measurements = np.zeros((10, 5))
        des = {'session': 0, 'subj': 0}
        obs_des = {'conds': np.array([0, 0, 1, 1, 2, 2, 2, 3, 4, 5])}
        chn_des = {'rois': np.array(['V1', 'V1', 'IT', 'IT', 'V4'])}
        data = rsd.Dataset(measurements=measurements,
                           descriptors=des,
                           obs_descriptors=obs_des,
                           channel_descriptors=chn_des
                           )
        splited_list = data.split_obs('conds')
        self.assertEqual(len(splited_list), 6)
        self.assertEqual(splited_list[0].n_obs, 2)
        self.assertEqual(splited_list[2].n_obs, 3)
        self.assertEqual(splited_list[0].n_channel, 5)
        self.assertEqual(splited_list[2].n_channel, 5)
        self.assertEqual(splited_list[2].obs_descriptors['conds'][0], 2)

    def test_dataset_split_channel(self):
        measurements = np.zeros((10, 5))
        des = {'session': 0, 'subj': 0}
        obs_des = {'conds': np.array([0, 0, 1, 1, 2, 2, 2, 3, 4, 5])}
        chn_des = {'rois': np.array(['V1', 'V1', 'IT', 'IT', 'V4'])}
        data = rsd.Dataset(measurements=measurements,
                           descriptors=des,
                           obs_descriptors=obs_des,
                           channel_descriptors=chn_des
                           )
        splited_list = data.split_channel('rois')
        self.assertEqual(len(splited_list), 3)
        self.assertEqual(splited_list[0].n_obs, 10)
        self.assertEqual(splited_list[2].n_obs, 10)
        self.assertEqual(splited_list[0].n_channel, 2)
        self.assertEqual(splited_list[2].n_channel, 1)
        self.assertEqual(splited_list[1].channel_descriptors['rois'][0], 'IT')

    def test_dataset_subset_obs(self):
        measurements = np.zeros((10, 5))
        des = {'session': 0, 'subj': 0}
        obs_des = {'conds': np.array([0, 0, 1, 1, 2, 2, 2, 3, 4, 5])}
        chn_des = {'rois': np.array(['V1', 'V1', 'IT', 'IT', 'V4'])}
        data = rsd.Dataset(measurements=measurements,
                           descriptors=des,
                           obs_descriptors=obs_des,
                           channel_descriptors=chn_des
                           )
        subset = data.subset_obs(by='conds', value=2)
        self.assertEqual(subset.n_obs, 3)
        self.assertEqual(subset.n_channel, 5)
        self.assertEqual(subset.obs_descriptors['conds'][0], 2)
        subset = data.subset_obs(by='conds', value=[2, 3])
        self.assertEqual(subset.n_obs, 4)
        self.assertEqual(subset.n_channel, 5)
        self.assertEqual(subset.obs_descriptors['conds'][0], 2)

    def test_dataset_subset_channel(self):
        measurements = np.zeros((10, 5))
        des = {'session': 0, 'subj': 0}
        obs_des = {'conds': np.array([0, 0, 1, 1, 2, 2, 2, 3, 4, 5])}
        chn_des = {'rois': np.array(['V1', 'V1', 'IT', 'IT', 'V4'])}
        data = rsd.Dataset(measurements=measurements,
                           descriptors=des,
                           obs_descriptors=obs_des,
                           channel_descriptors=chn_des
                           )
        subset = data.subset_channel(by='rois', value='IT')
        self.assertEqual(subset.n_obs, 10)
        self.assertEqual(subset.n_channel, 2)
        self.assertEqual(subset.channel_descriptors['rois'][0], 'IT')
        subset = data.subset_channel(by='rois', value=['IT', 'V4'])
        self.assertEqual(subset.n_obs, 10)
        self.assertEqual(subset.n_channel, 3)
        self.assertEqual(subset.channel_descriptors['rois'][0], 'IT')
        self.assertEqual(subset.channel_descriptors['rois'][-1], 'V4')


class TestDataComputations(unittest.TestCase):
    def setUp(self):
        measurements = np.random.rand(10, 5)
        des = {'session': 0, 'subj': 0}
        obs_des = {'conds': np.array([0, 0, 1, 1, 2, 2, 2, 3, 4, 5])}
        chn_des = {'rois': np.array(['V1', 'V1', 'IT', 'IT', 'V4'])}
        self.test_data = rsd.Dataset(
            measurements=measurements,
            descriptors=des,
            obs_descriptors=obs_des,
            channel_descriptors=chn_des
            )

    def test_average(self):
        avg = rsd.average_dataset(self.test_data)
        self.assertEqual(avg.shape, (5,))

    def test_average_by(self):
        avg, descriptor, n_obs = rsd.average_dataset_by(self.test_data, 'conds')
        self.assertEqual(avg.shape, (6, 5))
        self.assertEqual(len(descriptor), 6)
        self.assertEqual(descriptor[-1], 5)
        assert(np.all(self.test_data.measurements[-1] == avg[-1]))


class TestNoiseComputations(unittest.TestCase):
    def setUp(self):
        self.residuals = np.random.rand(100, 25)
        self.residuals = self.residuals - np.mean(self.residuals, axis=0,
                                                  keepdims=True)
        res_list = []
        for i in range(3):
            residuals = np.random.rand(100, 25)
            residuals = residuals - np.mean(residuals, axis=0, keepdims=True)
            res_list.append(residuals)
        self.res_list = res_list

    def test_cov(self):
        from pyrsa.data import cov_from_residuals
        cov = cov_from_residuals(self.residuals)

    def test_cov_list(self):
        from pyrsa.data import cov_from_residuals
        cov = cov_from_residuals(self.res_list)

    def test_prec(self):
        from pyrsa.data import prec_from_residuals
        cov = prec_from_residuals(self.residuals)

    def test_prec_list(self):
        from pyrsa.data import prec_from_residuals
        cov = prec_from_residuals(self.res_list)


class TestSave(unittest.TestCase):
    def test_dict_conversion(self):
        measurements = np.zeros((10, 5))
        des = {'session': 0, 'subj': 0}
        obs_des = {'conds': np.array([0, 0, 1, 1, 2, 2, 2, 3, 4, 5])}
        chn_des = {'rois': np.array(['V1', 'V1', 'IT', 'IT', 'V4'])}
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
        assert np.all(data_loaded.channel_descriptors['rois']
                      == chn_des['rois'])
        assert data_loaded.descriptors['subj'] == 0

    def test_save_load(self):
        import io
        f = io.BytesIO()  # Essentially a Mock file
        measurements = np.zeros((10, 5))
        des = {'session': 0, 'subj': 0}
        obs_des = {'conds': np.array([0, 0, 1, 1, 2, 2, 2, 3, 4, 5])}
        chn_des = {'rois': np.array(['V1', 'V1', 'IT', 'IT', 'V4'])}
        data = rsd.Dataset(measurements=measurements,
                           descriptors=des,
                           obs_descriptors=obs_des,
                           channel_descriptors=chn_des
                           )
        data.save(f, file_type='hdf5')
        data_loaded = rsd.load_dataset(f, file_type='hdf5')
        assert data_loaded.n_channel == data.n_channel
        assert np.all(data_loaded.obs_descriptors['conds'] == obs_des['conds'])
        assert np.all(data_loaded.channel_descriptors['rois']
                      == chn_des['rois'])
        assert data_loaded.descriptors['subj'] == 0

class TestMerge(unittest.TestCase):
    def setUp(self):
        # measurements = np.array([i for i in range(10,50)])
        # measurements = np.reshape(measurements, (4,10))
        measurements = np.random.rand(4, 10)
        des = {'session': 0, 'subj': 0}
        obs_des = {'conds': np.array([str(i) for i in range(1,5)])}
        chn_des = {'rois': np.array([chr(l) for l in range(65, 75)])}
        self.test_data = rsd.Dataset(
            measurements=measurements,
            descriptors=des,
            obs_descriptors=obs_des,
            channel_descriptors=chn_des
            )
    
    def test_merge(self):
        subsets = self.test_data.split_obs('conds')
        self.test_data_merged = rsd.merge_subsets(subsets)
        assert np.all(self.test_data_merged.measurements == \
                self.test_data.measurements)
        assert self.test_data_merged.descriptors == \
                self.test_data.descriptors
        assert np.all(self.test_data_merged.obs_descriptors['conds'] == \
                self.test_data.obs_descriptors['conds'])
        assert np.all(self.test_data_merged.channel_descriptors['rois'] == \
                self.test_data.channel_descriptors['rois'])

class TestOESplit(unittest.TestCase):
    def setUp(self):
        measurements = np.random.rand(4, 10)
        des = {'session': 0, 'subj': 0}
        chn_des = {'rois': np.array([chr(l) for l in range(65, 75)])}
        
        self.full_data = rsd.Dataset(
            measurements=measurements,
            descriptors=des,
            obs_descriptors={'conds': np.array([str(i) for i in range(1,5)])},
            channel_descriptors=chn_des
            ) 
        self.odd_data = rsd.Dataset(
            measurements=measurements[0::2],
            descriptors=des,
            obs_descriptors={'conds': np.array([str(i) for i in range(1,5,2)])},
            channel_descriptors=chn_des
            )         
        self.even_data = rsd.Dataset(
            measurements=measurements[1::2],
            descriptors=des,
            obs_descriptors={'conds': np.array([str(i) for i in range(2,5,2)])},
            channel_descriptors=chn_des
            )
    
    def test_odd_even_split(self):
        self.odd_split, self.even_split = rsd.odd_even_split(self.full_data, 'conds')
        
        assert np.all(self.odd_data.measurements == \
                self.odd_split.measurements)
        assert self.odd_data.descriptors == \
                self.odd_split.descriptors
        assert np.all(self.odd_data.obs_descriptors['conds'] == \
                self.odd_split.obs_descriptors['conds'])
        assert np.all(self.odd_data.channel_descriptors['rois'] == \
                self.odd_split.channel_descriptors['rois'])   
        assert np.all(self.even_data.measurements == \
                self.even_split.measurements)
        assert self.even_data.descriptors == \
                self.even_split.descriptors
        assert np.all(self.even_data.obs_descriptors['conds'] == \
                self.even_split.obs_descriptors['conds'])
        assert np.all(self.even_data.channel_descriptors['rois'] == \
                self.even_split.channel_descriptors['rois'])     

class TestNestedOESplit(unittest.TestCase):
    def setUp(self):
        measurements = np.random.rand(16, 10)
        des = {'session': 0, 'subj': 0}
        chn_des = {'rois': np.array([chr(l) for l in range(65, 75)])}
        conds =  np.array([str(i) for i in range(1,5)])
        runs = np.array([i for i in range(1,5)])
        self.full_data = rsd.Dataset(
            measurements=measurements,
            descriptors=des,
            obs_descriptors={'conds': np.hstack((conds, conds, conds, conds)),
                             'runs': np.repeat(runs, 4)},
            channel_descriptors=chn_des
            ) 
        self.odd_data = rsd.Dataset(
            measurements=np.append(measurements[0:4], measurements[8:12], axis=0),
            descriptors=des,
            obs_descriptors={'conds': np.hstack((conds, conds)),
                             'runs': np.repeat(runs[0::2], 4)},
            channel_descriptors=chn_des
            )         
        self.even_data = rsd.Dataset(
            measurements=np.append(measurements[4:8], measurements[12:16], axis=0),
            descriptors=des,
           obs_descriptors={'conds': np.hstack((conds, conds)),
                             'runs': np.repeat(runs[1::2], 4)},
            channel_descriptors=chn_des
            )
    
    def test_odd_even_split(self):
        self.odd_split, self.even_split = rsd.nested_odd_even_split(self.full_data, 'conds', 'runs')
        self.odd_split.sort_by('runs')
        self.even_split.sort_by('runs')
        
        assert np.all(self.odd_data.measurements == \
                self.odd_split.measurements)
        assert self.odd_data.descriptors == \
                self.odd_split.descriptors
        assert np.all(self.odd_data.obs_descriptors['conds'] == \
                self.odd_split.obs_descriptors['conds'])
        assert np.all(self.odd_data.channel_descriptors['rois'] == \
                self.odd_split.channel_descriptors['rois'])   
        assert np.all(self.even_data.measurements == \
                self.even_split.measurements)
        assert self.even_data.descriptors == \
                self.even_split.descriptors
        assert np.all(self.even_data.obs_descriptors['conds'] == \
                self.even_split.obs_descriptors['conds'])
        assert np.all(self.even_data.channel_descriptors['rois'] == \
                self.even_split.channel_descriptors['rois'])
            
if __name__ == '__main__':
    unittest.main()
