#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_data
Test for Dataset class
@author: baihan, jdiedrichsen, adkipnis
"""

import unittest
import rsatoolbox.data as rsd
import numpy as np
from numpy.testing import assert_array_equal


class TestData(unittest.TestCase):

    def setUp(self) -> None:
        self.rng = np.random.default_rng(0)
        return super().setUp()

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
        self.assertEqual(splited_list[0].descriptors.get('rois'), 'V1')
        self.assertEqual(splited_list[1].descriptors.get('rois'), 'IT')

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

    def test_copy(self):
        from rsatoolbox.data import Dataset
        orig = Dataset(
            measurements=self.rng.random((10, 5)),
            descriptors=dict(session=0, subj='AB'),
            obs_descriptors=dict(conds=np.array(
                [0, 0, 1, 1, 2, 2, 2, 3, 4, 5])),
            channel_descriptors=dict(
                rois=['V1', 'V1', 'IT', 'IT', 'V4'])
        )
        copy = orig.copy()
        # We don't want a reference:
        self.assertIsNot(copy, orig)
        self.assertIsNot(copy.measurements, orig.measurements)
        self.assertIsNot(
            copy.obs_descriptors.get('conds'),
            orig.obs_descriptors.get('conds')
        )
        # But check that attributes are equal
        assert_array_equal(copy.measurements, orig.measurements)
        self.assertEqual(copy.descriptors, orig.descriptors)
        assert_array_equal(
            copy.obs_descriptors.get('conds'),
            orig.obs_descriptors.get('conds')
        )
        assert_array_equal(
            copy.channel_descriptors.get('rois'),
            orig.channel_descriptors.get('rois')
        )

    def test_equality(self):
        from rsatoolbox.data import Dataset
        orig = Dataset(
            measurements=self.rng.random((10, 5)),
            descriptors=dict(session=0, subj='AB'),
            obs_descriptors=dict(conds=np.array(
                [0, 0, 1, 1, 2, 2, 2, 3, 4, 5])),
            channel_descriptors=dict(
                rois=['V1', 'V1', 'IT', 'IT', 'V4'])
        )
        other = orig.copy()
        self.assertEqual(orig, other)
        other = orig.copy()
        other.measurements[1, 1] = 1.1
        self.assertNotEqual(orig, other)
        other = orig.copy()
        other.obs_descriptors['conds'][1] = 9
        self.assertNotEqual(orig, other)
        other = orig.copy()
        other.channel_descriptors['rois'][1] = 'MT'
        self.assertNotEqual(orig, other)


class TestTemporalDataset(unittest.TestCase):

    def setUp(self) -> None:
        self.rng = np.random.default_rng(0)
        return super().setUp()

    def test_temporaldataset_simple_init(self):
        measurements = np.zeros((10, 5, 15))
        data = rsd.TemporalDataset(measurements)
        self.assertEqual(data.n_obs, 10)
        self.assertEqual(data.n_channel, 5)
        self.assertEqual(data.n_time, 15)
        self.assertEqual(len(data.time_descriptors['time']), 15)
        self.assertEqual(data.time_descriptors['time'][0], 0)

    def test_temporaldataset_full_init(self):
        measurements = np.zeros((10, 5, 15))
        des = {'session': 0, 'subj': 0}
        obs_des = {'conds': np.array(['cond_' + str(x)
                                      for x in np.arange(10)])}
        chn_des = {'rois': np.array(['roi_' + str(x) for x in np.arange(5)])}
        tim_des = {'time': np.linspace(0, 1000, 15)}
        data = rsd.TemporalDataset(measurements=measurements,
                                   descriptors=des,
                                   obs_descriptors=obs_des,
                                   channel_descriptors=chn_des,
                                   time_descriptors=tim_des
                                   )
        self.assertEqual(data.n_obs, 10)
        self.assertEqual(data.n_channel, 5)
        self.assertEqual(data.n_time, 15)
        self.assertEqual(data.descriptors, des)
        self.assertEqual(data.obs_descriptors, obs_des)
        self.assertEqual(data.channel_descriptors, chn_des)
        self.assertEqual(data.time_descriptors, tim_des)

    def test_temporaldataset_split_obs(self):
        measurements = np.zeros((10, 5, 15))
        des = {'session': 0, 'subj': 0}
        obs_des = {'conds': np.array([0, 0, 1, 1, 2, 2, 2, 3, 4, 5])}
        chn_des = {'rois': np.array(['V1', 'V1', 'IT', 'IT', 'V4'])}
        tim_des = {'time': np.linspace(0, 1000, 15)}
        data = rsd.TemporalDataset(measurements=measurements,
                                   descriptors=des,
                                   obs_descriptors=obs_des,
                                   channel_descriptors=chn_des,
                                   time_descriptors=tim_des
                                   )
        splited_list = data.split_obs('conds')
        self.assertEqual(len(splited_list), 6)
        self.assertEqual(splited_list[0].n_obs, 2)
        self.assertEqual(splited_list[2].n_obs, 3)
        self.assertEqual(splited_list[0].n_channel, 5)
        self.assertEqual(splited_list[2].n_channel, 5)
        self.assertEqual(splited_list[0].n_time, 15)
        self.assertEqual(splited_list[2].n_time, 15)
        self.assertEqual(splited_list[2].obs_descriptors['conds'][0], 2)

    def test_temporaldataset_split_channel(self):
        measurements = np.zeros((10, 5, 15))
        des = {'session': 0, 'subj': 0}
        obs_des = {'conds': np.array([0, 0, 1, 1, 2, 2, 2, 3, 4, 5])}
        chn_des = {'rois': np.array(['V1', 'V1', 'IT', 'IT', 'V4'])}
        tim_des = {'time': np.linspace(0, 1000, 15)}
        data = rsd.TemporalDataset(measurements=measurements,
                                   descriptors=des,
                                   obs_descriptors=obs_des,
                                   channel_descriptors=chn_des,
                                   time_descriptors=tim_des
                                   )
        splited_list = data.split_channel('rois')
        self.assertEqual(len(splited_list), 3)
        self.assertEqual(splited_list[0].n_obs, 10)
        self.assertEqual(splited_list[2].n_obs, 10)
        self.assertEqual(splited_list[0].n_channel, 2)
        self.assertEqual(splited_list[2].n_channel, 1)
        self.assertEqual(splited_list[0].n_time, 15)
        self.assertEqual(splited_list[2].n_time, 15)
        self.assertEqual(splited_list[1].channel_descriptors['rois'][0], 'IT')
        self.assertEqual(splited_list[1].descriptors['rois'], 'IT')

    def test_temporaldataset_split_time(self):
        measurements = np.zeros((10, 5, 15))
        des = {'session': 0, 'subj': 0}
        obs_des = {'conds': np.array([0, 0, 1, 1, 2, 2, 2, 3, 4, 5])}
        chn_des = {'rois': np.array(['V1', 'V1', 'IT', 'IT', 'V4'])}
        tim_des = {'time': np.linspace(0, 1000, 15)}
        data = rsd.TemporalDataset(measurements=measurements,
                                   descriptors=des,
                                   obs_descriptors=obs_des,
                                   channel_descriptors=chn_des,
                                   time_descriptors=tim_des
                                   )
        splited_list = data.split_time('time')
        self.assertEqual(len(splited_list), 15)
        self.assertEqual(splited_list[0].n_obs, 10)
        self.assertEqual(splited_list[2].n_obs, 10)
        self.assertEqual(splited_list[0].n_channel, 5)
        self.assertEqual(splited_list[2].n_channel, 5)
        self.assertEqual(splited_list[0].n_time, 1)
        self.assertEqual(splited_list[2].n_time, 1)
        self.assertEqual(
            splited_list[1].time_descriptors['time'][0], tim_des['time'][1])

    def test_temporaldataset_bin_time(self):
        measurements = self.rng.random((10, 5, 15))
        des = {'session': 0, 'subj': 0}
        obs_des = {'conds': np.array([0, 0, 1, 1, 2, 2, 2, 3, 4, 5])}
        chn_des = {'rois': np.array(['V1', 'V1', 'IT', 'IT', 'V4'])}
        tim_des = {'time': np.linspace(0, 1000, 15)}
        data = rsd.TemporalDataset(measurements=measurements,
                                   descriptors=des,
                                   obs_descriptors=obs_des,
                                   channel_descriptors=chn_des,
                                   time_descriptors=tim_des
                                   )
        bins = np.reshape(tim_des['time'], [5, 3])
        binned_data = data.bin_time('time', bins)
        self.assertEqual(binned_data.n_obs, 10)
        self.assertEqual(binned_data.n_channel, 5)
        self.assertEqual(binned_data.n_time, 5)
        self.assertEqual(
            binned_data.time_descriptors['time'][0], np.mean(bins[0]))
        self.assertEqual(binned_data.measurements[0, 0, 0], np.mean(
            measurements[0, 0, :3]))

    def test_temporaldataset_subset_obs(self):
        measurements = np.zeros((10, 5, 15))
        des = {'session': 0, 'subj': 0}
        obs_des = {'conds': np.array([0, 0, 1, 1, 2, 2, 2, 3, 4, 5])}
        chn_des = {'rois': np.array(['V1', 'V1', 'IT', 'IT', 'V4'])}
        tim_des = {'time': np.linspace(0, 1000, 15)}
        data = rsd.TemporalDataset(measurements=measurements,
                                   descriptors=des,
                                   obs_descriptors=obs_des,
                                   channel_descriptors=chn_des,
                                   time_descriptors=tim_des
                                   )
        subset = data.subset_obs(by='conds', value=2)
        self.assertEqual(subset.n_obs, 3)
        self.assertEqual(subset.n_channel, 5)
        self.assertEqual(subset.n_time, 15)
        self.assertEqual(subset.obs_descriptors['conds'][0], 2)
        subset = data.subset_obs(by='conds', value=[2, 3])
        self.assertEqual(subset.n_obs, 4)
        self.assertEqual(subset.n_channel, 5)
        self.assertEqual(subset.n_time, 15)
        self.assertEqual(subset.obs_descriptors['conds'][0], 2)

    def test_temporaldataset_subset_channel(self):
        measurements = np.zeros((10, 5, 15))
        des = {'session': 0, 'subj': 0}
        obs_des = {'conds': np.array([0, 0, 1, 1, 2, 2, 2, 3, 4, 5])}
        chn_des = {'rois': np.array(['V1', 'V1', 'IT', 'IT', 'V4'])}
        tim_des = {'time': np.linspace(0, 1000, 15)}
        data = rsd.TemporalDataset(measurements=measurements,
                                   descriptors=des,
                                   obs_descriptors=obs_des,
                                   channel_descriptors=chn_des,
                                   time_descriptors=tim_des
                                   )
        subset = data.subset_channel(by='rois', value='IT')
        self.assertEqual(subset.n_obs, 10)
        self.assertEqual(subset.n_channel, 2)
        self.assertEqual(subset.n_time, 15)
        self.assertEqual(subset.channel_descriptors['rois'][0], 'IT')
        subset = data.subset_channel(by='rois', value=['IT', 'V4'])
        self.assertEqual(subset.n_obs, 10)
        self.assertEqual(subset.n_channel, 3)
        self.assertEqual(subset.n_time, 15)
        self.assertEqual(subset.channel_descriptors['rois'][0], 'IT')
        self.assertEqual(subset.channel_descriptors['rois'][-1], 'V4')

    def test_temporaldataset_subset_time(self):
        measurements = np.zeros((10, 5, 15))
        des = {'session': 0, 'subj': 0}
        obs_des = {'conds': np.array([0, 0, 1, 1, 2, 2, 2, 3, 4, 5])}
        chn_des = {'rois': np.array(['V1', 'V1', 'IT', 'IT', 'V4'])}
        tim_des = {'time': np.linspace(0, 1000, 15)}
        data = rsd.TemporalDataset(measurements=measurements,
                                   descriptors=des,
                                   obs_descriptors=obs_des,
                                   channel_descriptors=chn_des,
                                   time_descriptors=tim_des
                                   )
        subset = data.subset_time(by='time', t_from=tim_des['time'][3],
                                  t_to=tim_des['time'][3])
        self.assertEqual(subset.n_obs, 10)
        self.assertEqual(subset.n_channel, 5)
        self.assertEqual(subset.n_time, 1)
        self.assertEqual(
            subset.time_descriptors['time'][0], tim_des['time'][3])
        subset = data.subset_time(by='time', t_from=tim_des['time'][3],
                                  t_to=tim_des['time'][5])
        self.assertEqual(subset.n_obs, 10)
        self.assertEqual(subset.n_channel, 5)
        self.assertEqual(subset.n_time, 3)
        self.assertEqual(
            subset.time_descriptors['time'][0], tim_des['time'][3])
        self.assertEqual(
            subset.time_descriptors['time'][-1], tim_des['time'][5])

    def test_temporaldataset_convert_to_dataset(self):
        measurements = np.zeros((10, 5, 15))
        des = {'session': 0, 'subj': 0}
        obs_des = {'conds': np.array([0, 0, 1, 1, 2, 2, 2, 3, 4, 5])}
        chn_des = {'rois': np.array(['V1', 'V1', 'IT', 'IT', 'V4'])}
        tim_des = {'time': np.linspace(0, 1000, 15),
                   'time_formatted': ['%0.0f ms' % (x) for x in np.linspace(0, 1000, 15)]}

        data_temporal = rsd.TemporalDataset(measurements=measurements,
                                            descriptors=des,
                                            obs_descriptors=obs_des,
                                            channel_descriptors=chn_des,
                                            time_descriptors=tim_des
                                            )
        data = data_temporal.convert_to_dataset('time')
        self.assertEqual(data.n_obs, 150)
        self.assertEqual(data.n_channel, 5)
        self.assertEqual(len(data.obs_descriptors['time']), 150)
        self.assertEqual(data.obs_descriptors['time'][0], tim_des['time'][0])
        self.assertEqual(data.obs_descriptors['time'][10], tim_des['time'][1])
        self.assertEqual(
            data.obs_descriptors['time_formatted'][10], tim_des['time_formatted'][1])
        self.assertEqual(data.obs_descriptors['conds'][0], obs_des['conds'][0])
        self.assertEqual(data.obs_descriptors['conds'][1], obs_des['conds'][1])

    def test_temporaldataset_time_as_channels(self):
        from rsatoolbox.data.dataset import TemporalDataset
        measurements = np.zeros((3, 2, 4)) # 3 trials, 2 channels, 4 timepoints
        des = {'session': 0, 'subj': 0}
        obs_des = {'conds': np.array([0, 1, 1])}
        chn_des = {'electrode': np.array(['A1', 'B2'])}
        tim_des = {'time': np.linspace(0, 900, 4),
                   'time_formatted': ['%0.0f ms' % (x) for x in np.linspace(0, 900, 4)]}
        data_temporal = TemporalDataset(
            measurements=measurements,
            descriptors=des,
            obs_descriptors=obs_des,
            channel_descriptors=chn_des,
            time_descriptors=tim_des
        )
        data = data_temporal.time_as_channels()
        self.assertEqual(data.n_obs, 3)
        self.assertEqual(data.n_channel, 2*4)
        self.assertEqual(len(data.channel_descriptors['time']), 2*4)
        assert_array_equal(
            data.channel_descriptors['time'],
            np.concatenate([tim_des['time'], tim_des['time']])
        )
        assert_array_equal(
            data.channel_descriptors['time_formatted'],
            tim_des['time_formatted'] + tim_des['time_formatted']
        )
        self.assertEqual(len(data.channel_descriptors['electrode']), 2*4)
        assert_array_equal(
            data.channel_descriptors['electrode'],
            ['A1', 'A1', 'A1', 'A1', 'B2', 'B2', 'B2', 'B2']
        )

    def test_copy(self):
        from rsatoolbox.data import TemporalDataset
        tps = np.linspace(0, 1000, 3)
        orig = TemporalDataset(
            measurements=self.rng.random((5, 4, 3)),
            descriptors=dict(session=0, subj='AB'),
            obs_descriptors=dict(conds=np.arange(5)),
            channel_descriptors=dict(
                rois=['V1', 'V2', 'V3', 'IT']),
            time_descriptors=dict(
                time=tps,
                time_formatted=['%0.0f ms' % (x) for x in tps]
            )
        )
        copy = orig.copy()
        # We don't want a reference:
        self.assertIsNot(copy, orig)
        self.assertIsNot(copy.measurements, orig.measurements)
        self.assertIsNot(
            copy.time_descriptors.get('time_formatted'),
            orig.time_descriptors.get('time_formatted')
        )
        # But check that attributes are equal
        assert_array_equal(copy.measurements, orig.measurements)
        self.assertEqual(copy.descriptors, orig.descriptors)
        assert_array_equal(
            copy.time_descriptors.get('time'),
            orig.time_descriptors.get('time')
        )
        assert_array_equal(
            copy.time_descriptors.get('time_formatted'),
            orig.time_descriptors.get('time_formatted')
        )

    def test_equality(self):
        from rsatoolbox.data import TemporalDataset
        tps = np.linspace(0, 1000, 3)
        orig = TemporalDataset(
            measurements=self.rng.random((5, 4, 3)),
            descriptors=dict(session=0, subj='AB'),
            obs_descriptors=dict(conds=np.arange(5)),
            channel_descriptors=dict(
                rois=['V1', 'V2', 'V3', 'IT']),
            time_descriptors=dict(
                time=tps,
                time_formatted=['%0.0f ms' % (x) for x in tps]
            )
        )
        other = orig.copy()
        self.assertEqual(orig, other)
        other = orig.copy()
        other.measurements[1, 1, 1] = 1.1
        self.assertNotEqual(orig, other)
        other = orig.copy()
        other.obs_descriptors['conds'][1] = 9
        self.assertNotEqual(orig, other)
        other = orig.copy()
        other.time_descriptors['time'][1] = 99
        self.assertNotEqual(orig, other)
        other = orig.copy()
        other.time_descriptors['time_formatted'][1] = 'Wednesday'
        self.assertNotEqual(orig, other)


class TestDataComputations(unittest.TestCase):

    def setUp(self):
        self.rng = np.random.default_rng(0)
        measurements = self.rng.random((10, 5))
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
        avg, descriptor, n_obs = rsd.average_dataset_by(
            self.test_data, 'conds')
        self.assertEqual(avg.shape, (6, 5))
        self.assertEqual(len(descriptor), 6)
        self.assertEqual(descriptor[-1], 5)
        assert (np.all(self.test_data.measurements[-1] == avg[-1]))


class TestNoiseComputations(unittest.TestCase):

    def setUp(self):
        self.rng = np.random.default_rng(0)
        self.residuals = self.rng.random((100, 25))
        self.residuals = self.residuals - np.mean(self.residuals, axis=0,
                                                  keepdims=True)
        res_list = []
        for i in range(3):
            residuals = self.rng.random((100, 25))
            residuals = residuals - np.mean(residuals, axis=0, keepdims=True)
            res_list.append(residuals)
        self.res_list = res_list
        self.dataset = rsd.Dataset(
            self.residuals,
            obs_descriptors={'obs': np.repeat(np.arange(10), 10)})

    def test_cov(self):
        from rsatoolbox.data import cov_from_residuals
        cov = cov_from_residuals(self.residuals)
        np.testing.assert_equal(cov.shape, [25, 25])

    def test_cov_list(self):
        from rsatoolbox.data import cov_from_residuals
        cov = cov_from_residuals(self.res_list)
        assert len(cov) == 3
        np.testing.assert_equal(cov[0].shape, [25, 25])

    def test_prec(self):
        from rsatoolbox.data import prec_from_residuals
        cov = prec_from_residuals(self.residuals)
        np.testing.assert_equal(cov.shape, [25, 25])

    def test_prec_list(self):
        from rsatoolbox.data import prec_from_residuals
        cov = prec_from_residuals(self.res_list)
        assert len(cov) == 3
        np.testing.assert_equal(cov[0].shape, [25, 25])

    def test_unbalanced(self):
        from rsatoolbox.data import cov_from_unbalanced
        cov = cov_from_unbalanced(self.dataset, 'obs')
        np.testing.assert_equal(cov.shape, [25, 25])

    def test_dataset(self):
        from rsatoolbox.data import cov_from_measurements
        cov = cov_from_measurements(self.dataset, 'obs')
        np.testing.assert_equal(cov.shape, [25, 25])

    def test_equal(self):
        from rsatoolbox.data import cov_from_measurements, cov_from_unbalanced
        cov1 = cov_from_measurements(self.dataset, 'obs')
        cov2 = cov_from_unbalanced(self.dataset, 'obs')
        np.testing.assert_allclose(cov1, cov2)


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


class TestOESplit(unittest.TestCase):

    def setUp(self) -> None:
        self.rng = np.random.default_rng(0)
        return super().setUp()

    def test_oe_split(self):
        measurements = self.rng.random((4, 10))
        des = {'session': 0, 'subj': 0}
        chn_des = {'rois': np.array([chr(i) for i in range(65, 75)])}

        self.full_data = rsd.Dataset(
            measurements=measurements,
            descriptors=des,
            obs_descriptors={
                'conds': np.array([str(i) for i in range(1, 5)])},
            channel_descriptors=chn_des
        )
        self.odd_data = rsd.Dataset(
            measurements=measurements[0::2],
            descriptors=des,
            obs_descriptors={
                'conds': np.array([str(i) for i in range(1, 5, 2)])},
            channel_descriptors=chn_des
        )
        self.even_data = rsd.Dataset(
            measurements=measurements[1::2],
            descriptors=des,
            obs_descriptors={
                'conds': np.array([str(i) for i in range(2, 5, 2)])},
            channel_descriptors=chn_des
        )
        self.odd_split, self.even_split = \
            self.full_data.odd_even_split('conds')
        np.testing.assert_array_equal(
            self.odd_data.measurements,
            self.odd_split.measurements)
        self.assertEqual(self.odd_data.descriptors,
                         self.odd_split.descriptors)
        np.testing.assert_array_equal(
            self.odd_data.obs_descriptors['conds'],
            self.odd_split.obs_descriptors['conds'])
        np.testing.assert_array_equal(
            self.odd_data.channel_descriptors['rois'],
            self.odd_split.channel_descriptors['rois'])
        np.testing.assert_array_equal(
            self.even_data.measurements,
            self.even_split.measurements)
        self.assertEqual(self.even_data.descriptors,
                         self.even_split.descriptors)
        np.testing.assert_array_equal(
            self.even_data.obs_descriptors['conds'],
            self.even_split.obs_descriptors['conds'])
        np.testing.assert_array_equal(
            self.even_data.channel_descriptors['rois'],
            self.even_split.channel_descriptors['rois'])

    def test_odd_even_split_nested(self):
        measurements = self.rng.random((16, 10))
        des = {'session': 0, 'subj': 0}
        chn_des = {'rois': np.array([chr(i) for i in range(65, 75)])}
        conds = np.array([str(i) for i in range(1, 5)])
        runs = np.array([i for i in range(1, 5)])
        self.full_data = rsd.Dataset(
            measurements=measurements,
            descriptors=des,
            obs_descriptors={'conds': np.hstack((conds, conds, conds, conds)),
                             'runs': np.repeat(runs, 4)},
            channel_descriptors=chn_des
        )
        self.odd_data = rsd.Dataset(
            measurements=np.append(measurements[0:4], measurements[8:12],
                                   axis=0),
            descriptors=des,
            obs_descriptors={'conds': np.hstack((conds, conds)),
                             'runs': np.repeat(runs[0::2], 4)},
            channel_descriptors=chn_des
        )
        self.even_data = rsd.Dataset(
            measurements=np.append(measurements[4:8], measurements[12:16],
                                   axis=0),
            descriptors=des,
            obs_descriptors={'conds': np.hstack((conds, conds)),
                             'runs': np.repeat(runs[1::2], 4)},
            channel_descriptors=chn_des
        )
        self.odd_split, self.even_split = self.full_data.nested_odd_even_split(
            'conds', 'runs')
        self.odd_split.sort_by('runs')
        self.even_split.sort_by('runs')
        np.testing.assert_array_equal(
            self.odd_data.measurements,
            self.odd_split.measurements)
        self.assertEqual(self.odd_data.descriptors,
                         self.odd_split.descriptors)
        np.testing.assert_array_equal(
            self.odd_data.obs_descriptors['conds'],
            self.odd_split.obs_descriptors['conds'])
        np.testing.assert_array_equal(
            self.odd_data.channel_descriptors['rois'],
            self.odd_split.channel_descriptors['rois'])
        np.testing.assert_array_equal(
            self.even_data.measurements,
            self.even_split.measurements)
        self.assertEqual(self.even_data.descriptors,
                         self.even_split.descriptors)
        np.testing.assert_array_equal(
            self.even_data.obs_descriptors['conds'],
            self.even_split.obs_descriptors['conds'])
        np.testing.assert_array_equal(
            self.even_data.channel_descriptors['rois'],
            self.even_split.channel_descriptors['rois'])


if __name__ == '__main__':
    unittest.main()
