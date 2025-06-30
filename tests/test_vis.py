#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_data
Test for visualization class
@author: baihan
"""
from unittest import TestCase, skip
from unittest.mock import patch
import numpy as np
import rsatoolbox.rdm as rsr
import rsatoolbox.vis as rsv
from scipy.spatial.distance import pdist

@skip('Skip until fix in #444')
class TestMDS(TestCase):

    def setUp(self) -> None:
        self.rng = np.random.default_rng(0)
        return super().setUp()

    @patch('rsatoolbox.vis.scatter_plot.show_scatter')
    def test_vis_mds_output_shape_corresponds_to_inputs(self, show_scatter):
        from rsatoolbox.vis.scatter_plot import show_MDS
        dis = self.rng.random((8, 10))
        mes = "Euclidean"
        des = {"session": 0, "subj": 0}
        rdms = rsr.RDMs(dissimilarities=dis,
                        dissimilarity_measure=mes, descriptors=des)
        show_MDS(rdms)
        coords = show_scatter.call_args[0][1]
        self.assertEqual(coords.shape, (8, 5, 2))

    @patch('rsatoolbox.vis.scatter_plot.show_scatter')
    def test_vis_weighted_mds_output_shape_corresponds_to_inputs(self, show_scatter):
        from rsatoolbox.vis.scatter_plot import show_MDS
        dis = self.rng.random((8, 10))
        wes = self.rng.random((8, 10))
        mes = "Euclidean"
        des = {"session": 0, "subj": 0}
        rdms = rsr.RDMs(dissimilarities=dis,
                        dissimilarity_measure=mes, descriptors=des)
        show_MDS(rdms, weights=wes)
        coords = show_scatter.call_args[0][1]
        self.assertEqual(coords.shape, (8, 5, 2))

    @patch('rsatoolbox.vis.scatter_plot.show_scatter')
    def test_vis_weighted_mds_output_behaves_like_mds(self, show_scatter):
        """
        Fails stochastically as MDS solution not deterministic
        """
        from rsatoolbox.vis.scatter_plot import show_MDS
        dis = self.rng.random((8, 10))
        wes = np.ones((8, 10))
        mes = "Euclidean"
        des = {"session": 0, "subj": 0}
        rdms = rsr.RDMs(dissimilarities=dis,
                        dissimilarity_measure=mes, descriptors=des)
        show_MDS(rdms)
        mds_coords = show_scatter.call_args[0][1]
        show_MDS(rdms, weights=wes)
        wmds_coords = show_scatter.call_args[0][1]
        diff = pdist(mds_coords[0]) - pdist(wmds_coords[0])
        self.assertLess(abs(diff.mean()), 0.02)


class Test_Icon(TestCase):

    def setUp(self) -> None:
        self.rng = np.random.default_rng(0)
        return super().setUp()

    def test_Icon_no_error(self):
        import PIL
        from rsatoolbox.vis import Icon
        import matplotlib.pyplot as plt

        test_im = PIL.Image.fromarray(255 * self.rng.random((50, 100)))
        ic5 = Icon(
            image=test_im, color="red", border_width=5, make_square=True, resolution=20
        )
        ic5.plot(0.8, 0.8)
        ic = Icon(image=255 * self.rng.random((50, 100)), cmap="Blues")
        ax = plt.axes(label="test")
        ic.plot(0.5, 0.5, ax=ax)
        ic2 = Icon(image=test_im, color="black",
                   border_width=15, string="test")
        ic2.plot(0.8, 0.2, ax=ax, size=0.4)
        ic2.x_tick_label(0.5, 0.15, offset=7)
        ic2.y_tick_label(0.5, 0.25, offset=7)
        ic3 = Icon(image=test_im, color="red",
                   border_width=5, make_square=True)
        ic3.plot(0.2, 0.2, size=0.4)
        ic4 = Icon(string="test")
        ic4.plot(0.2, 0.8, size=0.4)
        ic4.x_tick_label(0.75, 0.15, offset=7)
        ic4.y_tick_label(0.75, 0.25, offset=17)
        self.assertEqual(ic2.image, test_im)

    def test_Icon_from_rdm(self):
        from rsatoolbox.vis import Icon
        from rsatoolbox.rdm import RDMs
        rdm = RDMs(self.rng.random((1, 190)))
        ic = Icon(rdm)
        self.assertEqual(ic.final_image.size[0], 100)


def _dummy_rdm():
    import PIL
    import matplotlib.markers
    from collections import defaultdict

    rng = np.random.default_rng(0)

    markers = list(matplotlib.markers.MarkerStyle('').markers.keys())
    images = np.meshgrid(
        np.linspace(0.5, 1.0, 50), np.linspace(
            0.5, 1.0, 30), np.linspace(0.5, 1.0, 3)
    )
    images = [
        this_image * this_ind / 4.0 for this_ind in range(4) for this_image in images
    ]
    images = [PIL.Image.fromarray(255 * this_image, "RGB")
              for this_image in images]
    names = [
        this_class + this_ex
        for this_class in ("a", "b", "c", "d")
        for this_ex in ("1", "2", "3")
    ]
    n_con = len(names)
    icons = defaultdict(list)
    for this_marker, this_image, this_name in zip(markers, images, names):
        icons["image"].append(rsv.Icon(image=this_image))
        icons["marker"].append(rsv.Icon(marker=this_marker, color=[0, 0, 0]))
        icons["string"].append(rsv.Icon(string=this_name))
        icons["text"].append(this_name)
    ROIs = ["X", "Y", "Z"]
    return rsr.concat(
        [
            rsr.RDMs(
                dissimilarities=rng.random(
                    (1, int((n_con - 1) * n_con / 2))),
                dissimilarity_measure="1-rho",
                rdm_descriptors=dict(ROI=this_roi),
                pattern_descriptors=icons,
            )
            for this_roi in ROIs
        ]
    )
