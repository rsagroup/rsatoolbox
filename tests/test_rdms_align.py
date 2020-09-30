"""Unit tests for aligning and averaging partial RDMs
"""
#pylint: disable=import-outside-toplevel
from unittest import TestCase
from numpy import array, nan, isnan
from numpy.testing import assert_almost_equal, assert_array_equal
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
        assert_array_equal(
            partial_rdms.descriptors.get('weights'),
            array([
                [  1,   4, nan,    9,  nan,  nan],
                [nan, nan, nan,   16,   25,   36],
            ])
        )

    def test_mean_no_weights(self):
        """RDMs.mean() returns an RDMs with the nan omitted mean of the rdms
        """
        from pyrsa.rdm.rdms import RDMs
        partial_rdms = RDMs(
            dissimilarities=array([
                [  1,   2, nan,   3, nan, nan],
                [  2,   1, nan,   4,   5,   6],
            ])
        )
        assert_almost_equal(
            partial_rdms.mean(weights=None).dissimilarities,
            array([[ 1.5,  1.5, nan, 3.5, 5, 6]])
        )

    def test_weighted_mean(self):
        """Weights passed or stored in a descriptor are used in average
        """
        from pyrsa.rdm.rdms import RDMs
        partial_rdms = RDMs(
            dissimilarities=array([
                [  1,   2, nan,   3, nan, nan],
                [  2,   1, nan,   4,   5,   6],
            ])
        )
        weights = array([
            [  1,   1, nan,   1, nan, nan],
            [  2,   2, nan,   2,   2,   2],
        ])
        assert_almost_equal(
            partial_rdms.mean(weights=weights).dissimilarities,
            array([[1.6667, 1.3333, nan, 3.6667, 5.0000, 6.0000]]),
            decimal=3
        )
        partial_rdms.rdm_descriptors['weights'] = weights
        assert_almost_equal(
            partial_rdms.mean(weights='stored').dissimilarities,
            array([[1.6667, 1.3333, nan, 3.6667, 5.0000, 6.0000]]),
            decimal=3
        )
