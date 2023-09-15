"""Tests BIDS Input/output
"""
from unittest import TestCase


class TestIoBids(TestCase):

    def test_BidsLayout_find_derivative_files(self):
        from rsatoolbox.io.bids import BidsLayout
