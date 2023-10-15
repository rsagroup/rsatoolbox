from unittest import TestCase
from unittest.mock import Mock
import numpy


class Test_rdm_plot_conf(TestCase):
    
    def test_nrow_ncolumn(self):
        from rsatoolbox.vis.rdm_plot import MultiRdmPlot
        rdms = Mock()
        rdms.n_cond = 3
        rdms.n_rdm = 10
        rdms.get_matrices.return_value = numpy.zeros([10, 3, 3])
        conf = MultiRdmPlot.from_show_rdm_args(
            rdms,
            rdm_descriptor='name',
            show_colorbar='figure',
            pattern_descriptor='image',
            num_pattern_groups=5,
            icon_spacing=.9
        )
        self.assertEqual(conf.n_column, 4)
        self.assertEqual(conf.n_row, 3)
