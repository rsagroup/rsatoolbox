"""Various tests for converting Datasets to pandas.DataFrame and reverse
"""
# pylint: disable=C0415 ## allow imports on test level
from unittest import TestCase
import numpy
import pandas
from numpy.testing import assert_array_equal


class DatasetToDataframeTests(TestCase):
    """Acceptance test for converting to a dataframe and back.
    """

    def test_dataset_to_dataframe_and_back(self):
        """Converting a Dataset to a dataframe, and then loading that dataframe
        as a DataSet should preserve data and some metadata.
        """
        from rsatoolbox.data.dataset import Dataset
        ds_in = Dataset(
            measurements=numpy.random.rand(3, 2),
            descriptors=dict(foo='bar'),
            obs_descriptors=dict(participant=['a', 'b', 'c']),
            channel_descriptors=dict(name=['x', 'y'])
        )
        df = ds_in.to_df()
        assert_array_equal(df.x.values, ds_in.measurements[:, 0])
        self.assertEqual(
            df.columns.values.tolist(),
            ['x', 'y', 'participant', 'foo']
        )
        ds_out = Dataset.from_df(df)
        assert_array_equal(ds_out.measurements, ds_in.measurements)
        self.assertEqual(ds_out.descriptors, ds_in.descriptors)
        self.assertEqual(ds_out.obs_descriptors, ds_in.obs_descriptors)
        self.assertEqual(ds_out.channel_descriptors, ds_in.channel_descriptors)

    def test_dataset_to_dataframe_spec_channel_desc(self):
        """Converting a Dataset to a dataframe, and
        we specify the descriptors to use.
        """
        from rsatoolbox.data.dataset import Dataset
        ds_in = Dataset(
            measurements=numpy.random.rand(3, 2),
            descriptors=dict(foo='bar'),
            obs_descriptors=dict(participant=['a', 'b', 'c']),
            channel_descriptors=dict(foc=['x', 'y'], bac=[1, 2])
        )
        df = ds_in.to_df(channel_descriptor='bac')
        assert_array_equal(df[1].values, ds_in.measurements[:, 0])
        self.assertEqual(
            df.columns.values.tolist(),
            [1, 2, 'participant', 'foo']
        )

    def test_dataframe_to_dataset_spec_columns(self):
        """Creating a Dataset from a dataframe, and
        we specify the column roles.
        """
        from rsatoolbox.data.dataset import Dataset
        df = pandas.DataFrame([
            {'a': 1.1, 'b': 1, 3: 1.11, 'd': 'one', 'e': 1.111, 'f': 'bla'},
            {'a': 2.2, 'b': 2, 3: 2.22, 'd': 'two', 'e': 2.222, 'f': 'bla'},
            {'a': 3.3, 'b': 3, 3: 3.33, 'd': 'thr', 'e': 3.333, 'f': 'bla'},
            {'a': 4.4, 'b': 4, 3: 4.44, 'd': 'fou', 'e': 4.444, 'f': 'bla'},
        ])
        ds = Dataset.from_df(
            df,
            channels=['a', 'b', 3],
            channel_descriptor='foo'
        )
        assert_array_equal(ds.measurements, df[['a', 'b', 3]].values)
        self.assertEqual(ds.descriptors, dict(f='bla'))
        self.assertEqual(ds.obs_descriptors, dict(
            d=['one', 'two', 'thr', 'fou'],
            e=[1.111, 2.222, 3.333, 4.444]
        ))
        self.assertEqual(ds.channel_descriptors, dict(foo=['a', 'b', 3]))
