"""Unit tests for aligning and averaging partial RDMs
"""
#pylint: disable=import-outside-toplevel
from unittest import TestCase
from numpy import array, nan, isnan
from numpy.testing import assert_almost_equal
from scipy.stats import pearsonr


def non_nan(vectors, row=0):
    """Select only non-nan values from the first row"""
    return vectors[row, ~isnan(vectors[row, :])]


class RdmsAlignTests(TestCase):
    """Unit tests for aligning and averaging partial RDMs
    """

    def test_align(self):
        """The align method bring the RDMs as close together as possible
        """
        from pyrsa.rdm.rdms import RDMs
        partial=array([
            [  1,   2, nan,   3, nan, nan],
            [nan, nan, nan,   4,   5,   6],
        ])
        partial_rdms = RDMs(
            dissimilarities=partial
        )
        partial_rdms.align()
        aligned = partial_rdms.dissimilarities
        assert_almost_equal(aligned[0, 3], aligned[1, 3], decimal=4)
        assert_almost_equal(pearsonr(non_nan(partial), non_nan(aligned))[0], 1)
        matlab_aligned = array([
            [0.1438, 0.2877, nan, 0.4315,    nan,    nan],
            [   nan,    nan, nan, 0.4316, 0.5395, 0.6474]
        ])
        assert_almost_equal(aligned, matlab_aligned)

    def test_mean(self):
        """ _rdm_mean is separate function and method?? makes sense? or no use case?
        # _rdm_mean has "weighted" option
        # weighted by subset size is a third option
        """
