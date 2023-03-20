"""Tests for the merge operation on Datasets
"""
from __future__ import annotations
from unittest import TestCase
from numpy.testing import assert_array_equal


class MergeTests(TestCase):

    def test_merge_subsets(self):
        """Test the existing 'merge_subsets' interface
        """
        import numpy as np
        import rsatoolbox.data.dataset as rsd
        measurements = np.random.rand(4, 10)
        des = {'session': 0, 'subj': 0}
        obs_des = {'conds': np.array([str(i) for i in range(1, 5)])}
        chn_des = {'rois': np.array([chr(l) for l in range(65, 75)])}
        test_data = rsd.Dataset(
            measurements=measurements,
            descriptors=des,
            obs_descriptors=obs_des,
            channel_descriptors=chn_des
            )
        subsets = test_data.split_obs('conds')
        test_data_merged = rsd.merge_subsets(subsets)
        np.testing.assert_array_equal(
            test_data_merged.measurements,
            test_data.measurements)
        self.assertEqual(test_data_merged.descriptors,
                         test_data.descriptors)
        np.testing.assert_array_equal(
            test_data_merged.obs_descriptors['conds'],
            test_data.obs_descriptors['conds'])
        np.testing.assert_array_equal(
            test_data_merged.channel_descriptors['rois'],
            test_data.channel_descriptors['rois'])
        # TODO: deprecate old implementation
        
    def test_merge_datasets_standard(self):
        """Merge two standard datasets
        """
        from numpy.random import rand
        import numpy
        from rsatoolbox.data.dataset import Dataset
        from rsatoolbox.data.ops import merge_datasets
        ds1 = Dataset(
            measurements=rand(3, 2),
            descriptors=dict(foo='bar', same='us'),
            obs_descriptors=dict(cond=numpy.array(['a', 'b', 'c'])),
            channel_descriptors=dict(name=numpy.array(['x', 'y']))
        )
        ds2 = Dataset(
            measurements=rand(3, 2)+1,
            descriptors=dict(foo='baz', same='us'),
            obs_descriptors=dict(cond=numpy.array(['b', 'c', 'd'])),
            channel_descriptors=dict(name=numpy.array(['x', 'y']))
        )
        ds = merge_datasets([ds1, ds2])
        self.assertIsInstance(ds, Dataset)
        exp_meas = numpy.concatenate([
            ds1.measurements, ds2.measurements], axis=0)
        assert_array_equal(ds.measurements, exp_meas)
        assert_array_equal(
            ds.obs_descriptors.get('cond', []),
            ['a', 'b', 'c', 'b', 'c', 'd'])
        ## dataset descriptors that vary should become obs descriptor
        assert_array_equal(
            ds.obs_descriptors.get('foo', []),
            ['bar']*3 + ['baz']*3)
        ## dataset descriptors that are identical should remain
        self.assertEqual(ds.descriptors.get('same'), 'us')

    def test_merge_datasets_temporal(self):
        from numpy.random import rand
        import numpy
        from rsatoolbox.data.dataset import TemporalDataset
        from rsatoolbox.data.ops import merge_datasets
        ds1 = TemporalDataset(
            measurements=rand(3, 2, 4),
            time_descriptors=dict(time=numpy.array([0, 1, 2, 3])/10),
        )
        ds2 = TemporalDataset(
            measurements=rand(3, 2, 4)+1,
            time_descriptors=dict(time=numpy.array([0, 1, 2, 3])/10)
        )
        ds = merge_datasets([ds1, ds2])
        self.assertIsInstance(ds, TemporalDataset)
        exp_meas = numpy.concatenate([
            ds1.measurements, ds2.measurements], axis=0)
        assert_array_equal(ds.measurements, exp_meas)
        assert_array_equal(
            ds.time_descriptors.get('time', []),
            [0.0, 0.1, 0.2, 0.3])
