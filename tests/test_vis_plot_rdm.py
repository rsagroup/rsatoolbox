from unittest import TestCase
from tests.test_vis import _dummy_rdm


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
