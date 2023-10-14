from unittest import TestCase
from unittest.mock import Mock
import numpy


class Test_rdm_plot_conf(TestCase):
    
    def test_simple_conf(self):
        from rsatoolbox.vis.rdm_plot import MultiRdmPlotConf
        rdms = Mock()
        rdms.n_cond = 3
        rdms.n_rdm = 2
        rdms.get_matrices.return_value = numpy.zeros([1, 3, 3])
        conf = MultiRdmPlotConf.from_plot_rdm_args(
            rdms,
            rdm_descriptor='name',
            show_colorbar='figure',
            pattern_descriptor='image',
            num_pattern_groups=5,
            icon_spacing=.9
        )
