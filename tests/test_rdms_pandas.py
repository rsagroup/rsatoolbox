from __future__ import annotations
from unittest import TestCase
from typing import TYPE_CHECKING, Union, List
from numpy.testing import assert_array_equal
import numpy
from pandas import Series, DataFrame
if TYPE_CHECKING:
    from numpy.typing import NDArray


class RdmsToPandasTests(TestCase):

    def assertValuesEqual(self,
                          actual: Series,
                          expected: Union[NDArray, List]):
        assert_array_equal(numpy.asarray(actual.values), expected)

    def test_to_df(self):
        """Convert an RDMs object to a pandas DataFrame

        Default is long form; multiple rdms are stacked row-wise.
        """
        from rsatoolbox.rdm.rdms import RDMs
        dissimilarities = numpy.random.rand(2, 6)
        rdms = RDMs(
            dissimilarities,
            rdm_descriptors=dict(xy=[c for c in 'xy']),
            pattern_descriptors=dict(abcd=numpy.asarray([c for c in 'abcd']))
        )
        df = rdms.to_df()
        self.assertIsInstance(df, DataFrame)
        self.assertEqual(len(df.columns), 7)
        self.assertValuesEqual(df.dissimilarity, dissimilarities.ravel())
        self.assertValuesEqual(df['rdm_index'], ([0]*6) + ([1]*6))
        self.assertValuesEqual(df['xy'], (['x']*6) + (['y']*6))
        self.assertValuesEqual(df['pattern_index_1'],
                               ([0]*3 + [1]*2 + [2]*1)*2)
        self.assertValuesEqual(df['pattern_index_2'], [1, 2, 3, 2, 3, 3]*2)
        self.assertValuesEqual(df['abcd_1'], [c for c in 'aaabbc']*2)
        self.assertValuesEqual(df['abcd_2'], [c for c in 'bcdcdd']*2)
