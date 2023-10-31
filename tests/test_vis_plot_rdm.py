from unittest import TestCase
from tests.test_vis import _dummy_rdm
import numpy


class Test_rdm_plot(TestCase):
    """""
    "No Error" tests
    """

    @classmethod
    def setUpClass(cls):
        cls.rdm = _dummy_rdm()

    def setUp(self) -> None:
        import matplotlib.pyplot
        matplotlib.pyplot.close('all')
        return super().setUp()

    def test_show_rdm_no_arg_no_error(self):
        """regression test for crashes when gridlines is None (and needs to be set to []
        internally to avoid breaking mpl"""
        from rsatoolbox.vis.rdm_plot import show_rdm
        show_rdm(self.rdm[0])

    def test_show_rdm_text_label_no_error(self):
        """test RDM visualisation with vanilla Matplotlib text labels."""
        from rsatoolbox.vis.rdm_plot import show_rdm
        show_rdm(self.rdm[0], pattern_descriptor="index")

    def test_show_rdm_icon_image_no_error(self):
        from rsatoolbox.vis.rdm_plot import show_rdm
        show_rdm(self.rdm[0], pattern_descriptor="image")

    def test_show_rdm_icon_image_groups_no_error(self):
        from rsatoolbox.vis.rdm_plot import show_rdm
        show_rdm(self.rdm[0], pattern_descriptor="image", num_pattern_groups=4)

    def test_show_rdm_icon_marker_no_error(self):
        from rsatoolbox.vis.rdm_plot import show_rdm
        show_rdm(self.rdm[0], pattern_descriptor="marker")

    def test_show_rdm_icon_string_no_error(self):
        from rsatoolbox.vis.rdm_plot import show_rdm
        show_rdm(self.rdm[0], pattern_descriptor="string")

    def test_contour_coords(self):
        """Helper function determines edge coordinates for a 2d mask
        """
        from rsatoolbox.vis.rdm_plot import _contour_coords
        mask = numpy.array([
            [0, 0, 0, 0, 0],
            [0, 1, 1, 0, 0],
            [0, 0, 1, 1, 0],
            [0, 0, 1, 1, 0],
            [0, 0, 0, 0, 0],
        ])
        OFFSET = 0
        def with_offset(*args):
            return tuple([v+OFFSET for v in args])
        expected = [
            with_offset(1, 1, 2, 1),
            with_offset(2, 2, 1, 2),
            with_offset(1, 2, 1, 1),

            with_offset(2, 1, 3, 1),
            with_offset(3, 1, 3, 2),

            with_offset(2, 3, 2, 2), # 3rd pix, left side

            with_offset(3, 4, 2, 4), # pixel below that one, bottom
            with_offset(2, 4, 2, 3), # left

            with_offset(3, 2, 4, 2),
            with_offset(4, 2, 4, 3),

            with_offset(4, 3, 4, 4),
            with_offset(4, 4, 3, 4),
        ]
        actual = list(_contour_coords(mask, offset=OFFSET))
        self.assertEqual(actual, expected)
