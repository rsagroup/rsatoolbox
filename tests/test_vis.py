#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_data
Test for visualization class
@author: baihan
"""
import unittest
import numpy as np
import pyrsa.vis as rsv
import pyrsa.rdm as rsr
from scipy.spatial.distance import pdist


class TestVIS(unittest.TestCase):

    def test_vis_mds_output_shape_corresponds_to_inputs(self):
        dis = np.random.rand(8, 10)
        mes = "Euclidean"
        des = {'session': 0, 'subj': 0}
        rdms = rsr.RDMs(dissimilarities=dis,
                        dissimilarity_measure=mes,
                        descriptors=des)
        mds_emb = rsv.mds(rdms)
        self.assertEqual(mds_emb.shape, (8, 5, 2))

    def test_vis_3d_mds_output_shape_corresponds_to_inputs(self):
        dis = np.random.rand(8, 10)
        mes = "Euclidean"
        des = {'session': 0, 'subj': 0}
        rdms = rsr.RDMs(dissimilarities=dis,
                        dissimilarity_measure=mes,
                        descriptors=des)
        mds_emb = rsv.mds(rdms, dim=3)
        self.assertEqual(mds_emb.shape, (8, 5, 3))

    def test_vis_weighted_mds_output_shape_corresponds_to_inputs(self):
        dis = np.random.rand(8, 10)
        wes = np.random.random((8, 10))
        mes = "Euclidean"
        des = {'session': 0, 'subj': 0}
        rdms = rsr.RDMs(dissimilarities=dis,
                        dissimilarity_measure=mes,
                        descriptors=des)
        mds_emb = rsv.mds(rdms, weight=wes)
        self.assertEqual(mds_emb.shape, (8, 5, 2))

    def test_vis_3d_weighted_mds_output_shape_corresponds_to_inputs(self):
        dis = np.random.rand(8, 10)
        wes = np.random.random((8, 10))
        mes = "Euclidean"
        des = {'session': 0, 'subj': 0}
        rdms = rsr.RDMs(dissimilarities=dis,
                        dissimilarity_measure=mes,
                        descriptors=des)
        mds_emb = rsv.mds(rdms, dim=3, weight=wes)
        self.assertEqual(mds_emb.shape[0], 8)
        self.assertEqual(mds_emb.shape[1], 5)
        self.assertEqual(mds_emb.shape[2], 3)

    def test_vis_weighted_mds_output_behaves_like_mds(self):
        dis = np.random.rand(8, 10)
        wes = np.ones((8, 10))
        mes = "Euclidean"
        des = {'session': 0, 'subj': 0}
        rdms = rsr.RDMs(dissimilarities=dis,
                        dissimilarity_measure=mes,
                        descriptors=des)
        mds_emb = rsv.mds(rdms)
        wmds_emb = rsv.mds(rdms, weight=wes)
        np.testing.assert_allclose(pdist(mds_emb[0]), pdist(wmds_emb[0]),
                                   atol=3e-1)

    def test_vis_3d_weighted_mds_output_behaves_like_mds(self):
        dis = np.random.rand(8, 10)
        wes = np.ones((8, 10))
        mes = "Euclidean"
        des = {'session': 0, 'subj': 0}
        rdms = rsr.RDMs(dissimilarities=dis,
                        dissimilarity_measure=mes,
                        descriptors=des)
        mds_emb = rsv.mds(rdms, dim=3)
        wmds_emb = rsv.mds(rdms, dim=3, weight=wes)
        np.testing.assert_allclose(pdist(mds_emb[0]), pdist(wmds_emb[0]),
                                   atol=3e-1)


class Test_Icon(unittest.TestCase):

    def test_Icon_no_error(self):
        import PIL
        from pyrsa.vis import Icon
        import matplotlib.pyplot as plt
        test_im = PIL.Image.fromarray(255 * np.random.rand(50, 100))
        ic5 = Icon(image=test_im, col='red', border_width=5,
                   make_square=True, resolution=20)
        ic5.plot(0.8, 0.8)
        ic = Icon(image=255 * np.random.rand(50, 100), cmap='Blues')
        ax = plt.axes(label='test')
        ic.plot(0.5, 0.5, ax=ax)
        ic2 = Icon(image=test_im, col='black', border_width=15,
                   string='test')
        ic2.plot(0.8, 0.2, ax=ax, size=0.4)
        ic2.x_tick_label(0.5, 0.15, offset=7)
        ic2.y_tick_label(0.5, 0.25, offset=7)
        ic3 = Icon(image=test_im, col='red', border_width=5,
                   make_square=True)
        ic3.plot(0.2, 0.2, size=0.4)
        ic4 = Icon(string='test')
        ic4.plot(0.2, 0.8, size=0.4)
        ic4.x_tick_label(0.75, 0.15, offset=7)
        ic4.y_tick_label(0.75, 0.25, offset=17)
        self.assertEqual(ic2.image, test_im)

    def test_Icon_from_rdm(self):
        from pyrsa.vis import Icon
        from pyrsa.rdm import RDMs
        rdm = RDMs(np.random.rand(1, 190))
        ic = Icon(rdm)
        self.assertEqual(ic.final_image.size[0], 100)


class Test_model_plot(unittest.TestCase):

    def test_y_label(self):
        from pyrsa.vis.model_plot import _get_y_label
        y_label = _get_y_label('corr')
        self.assertIsInstance(y_label, str)

    def test_descr(self):
        from pyrsa.vis.model_plot import _get_model_comp_descr
        descr = _get_model_comp_descr(
            't-test', 5, 'fwer', 0.05, 1000,
            'boostrap_rdm', 'ci56', 'droplets', 'icicles')
        self.assertIsInstance(descr, str)


if __name__ == '__main__':
    unittest.main()
