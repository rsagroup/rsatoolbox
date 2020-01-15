#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from unittest import TestCase


class TestDescriptorUtils(TestCase):

    def test_format_descriptor(self):
        from pyrsa.util.descriptor_utils import format_descriptor
        descriptors = {'foo': 'bar', 'foz': 12.3}
        self.assertEqual(
            format_descriptor(descriptors),
            'foo = bar\nfoz = 12.3\n'
        )

    def test_parse_input_descriptor(self):
        from pyrsa.util.descriptor_utils import parse_input_descriptor
        descriptors = {'foo': 'bar', 'foz': 12.3}
        self.assertEqual(
            parse_input_descriptor(descriptors),
            descriptors
        )
        self.assertEqual(
            parse_input_descriptor(None),
            {}
        )

    def test_check_descriptor_length(self):
        from pyrsa.util.descriptor_utils import check_descriptor_length
        descriptors = {'foo': ['bar', 'bar2']}
        assert check_descriptor_length(descriptors, 2)
        assert not check_descriptor_length(descriptors, 3)
        descriptors = {'foo': ['bar']}
        assert check_descriptor_length(descriptors, 1)

    def test_subset_descriptor(self):
        import numpy as np
        from pyrsa.util.descriptor_utils import subset_descriptor
        descriptors = {'foo': ['bar', 'bar2']}
        self.assertEqual(
                subset_descriptor(descriptors,0),
                {'foo': ['bar']}
                )
        self.assertEqual(
                subset_descriptor(descriptors,np.array([True,False])),
                {'foo': ['bar']}
                )
        self.assertEqual(
                subset_descriptor(descriptors,(0,1)),
                {'foo': ['bar', 'bar2']}
                )

    def test_check_descriptor_length_error(self):
        from pyrsa.util.descriptor_utils import check_descriptor_length_error
        descriptors = {'foo': ['bar', 'bar2']}
        check_descriptor_length_error(descriptors,'test',2)
