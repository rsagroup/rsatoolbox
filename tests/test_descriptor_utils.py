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
