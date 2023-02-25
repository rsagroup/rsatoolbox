from unittest import TestCase
from numpy.testing import assert_array_equal
import numpy
import pandas


class RdmsToPandasTests(TestCase):

    def test_to_df(self):
        """Convert an RDMs object to a pandas DataFrame

        Default is long form; multiple rdms are stacked row-wise.
        """
        from rsatoolbox.rdm.rdms import RDMs
        dissimilarities = numpy.random.rand(2, 3)
        conds = [c for c in 'abc']
        rdms = RDMs(
            dissimilarities,
            rdm_descriptors=dict(xy=['x', 'y']),
            pattern_descriptors=dict(abc=numpy.asarray(conds))
        )
        df = rdms.to_df()
        self.assertIsInstance(df, pandas.DataFrame)
        self.assertEqual(len(df.columns), 5)
        assert_array_equal(df.dissimilarity.values, dissimilarities.ravel())
        assert_array_equal(df['rdm_index'].values, ([0]*3) + ([1]*3)) assert_array_equal(df['rdm_xy'].values, (['x']*3) + (['y']*3))
        assert_array_equal(df['pattern_index'].values, list(range(3))*2)
        assert_array_equal(df['pat_abc'].values, conds*2)
