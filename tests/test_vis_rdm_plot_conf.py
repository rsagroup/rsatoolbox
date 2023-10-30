from unittest import TestCase
from unittest.mock import Mock
from numpy.testing import assert_array_equal
import numpy


class TestRdmPlot(TestCase):

    def test_from_show_rdm_args__nrow_ncolumn(self):
        from rsatoolbox.vis.rdm_plot import MultiRdmPlot, Symmetry
        rdms = Mock()
        rdms.n_cond = 3
        rdms.n_rdm = 10
        rdms.get_matrices.return_value = numpy.zeros([10, 3, 3])
        conf = MultiRdmPlot.from_show_rdm_args(
            rdms,
            pattern_descriptor = None,
            cmap = 'bone',
            rdm_descriptor=None,
            n_column = None,
            n_row = None,
            show_colorbar = 'figure',
            gridlines = None,
            num_pattern_groups = None,
            figsize = None,
            nanmask = "diagonal",
            style = None,
            vmin = None,
            vmax = None,
            icon_spacing = 1.0,
            linewidth = 0.5,
            overlay = None,
            overlay_color='#00ff0050',
            overlay_symmetry=Symmetry.UPPER,
            contour = None,
            contour_color = 'red',
            contour_symmetry=Symmetry.BOTH
        )
        self.assertEqual(conf.n_column, 4)
        self.assertEqual(conf.n_row, 3)

    def test_from_show_rdm_args__multi_contour_overlay(self):
        from rsatoolbox.vis.rdm_plot import MultiRdmPlot, Symmetry
        rdms = Mock()
        rdms.n_cond = 3
        rdms.n_rdm = 10
        rdms.get_matrices.return_value = numpy.zeros([10, 3, 3])
        mask_rdm = Mock()
        mask_rdm.dissimilarities = numpy.array([[0, 1, 0, 1, 0, 1]])
        rdms.subset.return_value = mask_rdm
        rdms.dissimilarities = numpy.zeros([10, 6])
        conf = MultiRdmPlot.from_show_rdm_args(
            rdms,
            pattern_descriptor = None,
            cmap = 'bone',
            rdm_descriptor=None,
            n_column = None,
            n_row = None,
            show_colorbar = 'figure',
            gridlines = None,
            num_pattern_groups = None,
            figsize = None,
            nanmask = "diagonal",
            style = None,
            vmin = None,
            vmax = None,
            icon_spacing = 1.0,
            linewidth = 0.5,
            overlay = numpy.array([0,0,0,1,1,1]),
            overlay_color='#00ff0050',
            overlay_symmetry=Symmetry.BOTH,
            contour = ('name', 'foo'),
            contour_color = 'red',
            contour_symmetry=Symmetry.LOWER
        )
        assert_array_equal(conf.overlay, numpy.array([0, 0, 0, 1, 1, 1]))
        assert_array_equal(conf.contour, numpy.array([0, 1, 0, 1, 0, 1]))

    def test_single_from_show_rdm_panel_args(self):
        from rsatoolbox.vis.rdm_plot import SingleRdmPlot, Symmetry
        rdms = Mock()
        rdms.n_cond = 3
        rdms.n_rdm = 1
        rdms.get_matrices.return_value = numpy.zeros([1, 3, 3])
        rdms.rdm_descriptors = dict()
        mask_rdm = Mock()
        mask_rdm.dissimilarities = numpy.array([[0, 1, 0, 1, 0, 1]])
        rdms.dissimilarities = numpy.zeros([10, 6])
        conf = SingleRdmPlot.from_show_rdm_panel_args(
            rdms = rdms,
            cmap = 'bone',
            nanmask = None,
            rdm_descriptor = None,
            gridlines = None,
            vmin = None,
            vmax = None,
            overlay = numpy.array([0, 0, 0, 1, 1, 1]),
            overlay_color='#00ff0050',
            overlay_symmetry=Symmetry.UPPER,
            contour = numpy.array([0, 1, 0, 1, 0, 1]),
            contour_color = 'red',
            contour_symmetry=Symmetry.LOWER
        )
        assert_array_equal(conf.overlay, numpy.array([0, 0, 0, 1, 1, 1]))
        assert_array_equal(conf.contour, numpy.array([0, 1, 0, 1, 0, 1]))
