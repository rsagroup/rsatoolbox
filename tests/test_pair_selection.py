from __future__ import annotations
from unittest import TestCase
from pandas.testing import assert_frame_equal
from pandas import DataFrame
from numpy import array
from scipy.spatial.distance import squareform


class PairSelectionTests(TestCase):

    def test_percentile_with_target(self):
        """From pairs which include the target pattern,
        return those that fall between the given percentiles.
        """
        from rsatoolbox.rdm.rdms import RDMs
        from rsatoolbox.rdm.pairs import pairs_by_percentile
        rdms = RDMs(
            dissimilarities=squareform(array([
                [0, 7, 5, 4, 3],
                [7, 0, 6, 5, 4],
                [5, 6, 0, 9, 9],
                [4, 5, 9, 0, 9],
                [3, 4, 9, 9, 0],
            ])),
            pattern_descriptors=dict(cond=['a', 'b', 'c', 'd', 'e'])
        )
        ## 25% lowest dissimilarities
        out = pairs_by_percentile(rdms, max=25, with_pattern=dict(cond='a'))
        assert_frame_equal(out, DataFrame([
            dict(cond='e', dissim=3),
        ]))
        ## 40% - 80% mid range dissimilarities
        out = pairs_by_percentile(rdms, min=40, max=70, with_pattern=dict(cond='b'))
        assert_frame_equal(out, DataFrame([
            dict(cond='c', dissim=6),
            dict(cond='d', dissim=5),
        ]))
