"""
Tests the port of the matlab colormap

@author: iancharest
"""
from unittest import TestCase
import numpy as np


class ColorTests(TestCase):

    def test_color_scale(self):
        from rsatoolbox.vis.colors import color_scale
        n_cols = 10
        cols = color_scale(n_cols)
        n_cols_returned, n_rgb = cols.shape
        self.assertEqual(n_cols_returned, n_cols)
        self.assertEqual(n_rgb, 3)

    def test_rdm_colormap(self):
        from rsatoolbox.vis.colors import rdm_colormap_classic
        n_cols = 10
        cols = rdm_colormap_classic(n_cols)
        n_cols_returned, n_rgb = cols.colors.shape
        last_color = [1., 1., 0]
        self.assertEqual(n_cols_returned, n_cols)
        self.assertEqual(n_rgb, 3)
        np.testing.assert_array_almost_equal(last_color, cols.colors[-1])
