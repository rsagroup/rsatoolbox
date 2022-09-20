from __future__ import annotations
from unittest import TestCase
from pandas.util.testing import assert_frame_equal
import numpy
import pandas


class PairSelectionTests(TestCase):

    def test_percentile_with_target(self):
        """From pairs which include the target pattern,
        return those that fall under the given percentile.
        """
        from rsatoolbox.rdm.rdms import RDMs
        from rsatoolbox.rdm.pairs import select_pairs_by_percentile
        rdms = RDMs()
        select_pairs_by_percentile(group_rdm, -67, with_pattern=dict())
        assert_frame_equal()
