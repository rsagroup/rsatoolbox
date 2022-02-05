from unittest import TestCase
import numpy
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
        df = ds_in.to_DataFrame()
        assert_array_equal(df.x.values, ds_in.measurements[:, 0])
        self.assertEqual(
            df.columns.values.tolist(),
            ['x', 'y', 'participant', 'foo']
        )
        ds_out = Dataset.from_DataFrame(df)
        assert_array_equal(ds_out.measurements, ds_in.measurements)
        self.assertEqual(ds_out.measurements, ds_in.measurements)
        self.assertEqual(ds_out.descriptors, ds_in.descriptors)
        self.assertEqual(ds_out.obs_descriptors, ds_in.obs_descriptors)
        self.assertEqual(ds_out.channel_descriptors, ds_in.channel_descriptors)
