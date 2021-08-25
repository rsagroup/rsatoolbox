#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_colors
Test for colors
@author: iancharest
"""

import unittest
import numpy as np
from rsatoolbox.vis.colors import color_scale
from rsatoolbox.vis.colors import rdm_colormap


class ColorTests(unittest.TestCase):

    def test_color_scale(self):
        n_cols = 10
        cols = color_scale(n_cols)
        n_cols_returned, n_rgb = cols.shape
        self.assertEqual(n_cols_returned, n_cols)
        self.assertEqual(n_rgb, 3)

    def test_rdm_colormap(self):
        n_cols = 10
        cols = rdm_colormap(n_cols)
        n_cols_returned, n_rgb = cols.colors.shape
        last_color = [1., 1., 0]
        self.assertEqual(n_cols_returned, n_cols)
        self.assertEqual(n_rgb, 3)
        np.testing.assert_array_almost_equal(last_color, cols.colors[-1])
