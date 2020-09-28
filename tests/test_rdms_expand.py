"""Unit tests for expanding RDMs to include missing data.
"""
#pylint: disable=import-outside-toplevel
from unittest import TestCase
import numpy
from numpy.testing import assert_array_equal


class RdmsExpandTests(TestCase):
    """Unit tests for expanding RDMs to include missing data.
    """

    def test_expand_based_on_list_of_rdms_objects(self):
        """In this case the complete list of conditions is determined
        from the RDMs passed
        """
        from pyrsa.rdm.rdms import RDMs
        rdms1 = RDMs(
            dissimilarities='',
            dissimilarity_measure='shared_measure',
            descriptors=dict(shared_desc='shared_val', diff_desc='one'),
            rdm_descriptors=dict(rdesc=['foo1']),
            pattern_descriptors=dict(cond=['a', 'b', 'c']),
        )
        rdms23 = RDMs(
            dissimilarities='',
            dissimilarity_measure='shared_measure',
            descriptors=dict(shared_desc='shared_val', diff_desc='two-three'),
            rdm_descriptors=dict(rdesc=['foo2', 'foo3']),
            pattern_descriptors=dict(cond=['b', 'c', 'd']),
        )
        rdms = RDMs.expand([rdms1, rdms23])
        self.assertEqual(rdms.n_rdm, 3)
        self.assertEqual(rdms.n_cond, 9)
        self.assertEqual(rdms.dissimilarity_measure, 'shared_measure')
        self.assertEqual(rdms.descriptors.get('shared_desc'), 'shared_val')
        assert_array_equal(
            rdms.rdm_descriptors.get('diff_desc'),
            ['one', 'two-three', 'two-three']
        )
        assert_array_equal(
            rdms.rdm_descriptors.get('rdesc'),
            ['foo1', 'foo2', 'foo3']
        )
        assert_array_equal(
            rdms.pattern_descriptors.get('cond'),
            ['a', 'b', 'c', 'd']
        )
        assert_array_equal(
            rdms.dissimilarities,
            numpy.array()
        )
