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


class TestVIS(unittest.TestCase):

    def test_vis_mds_output_shape_corresponds_to_inputs(self):
        dis = np.random.rand(8, 10)
        mes = "Euclidean"
        des = {'session': 0, 'subj': 0}
        rdms = rsr.RDMs(dissimilarities=dis,
                        dissimilarity_measure=mes,
                        descriptors=des)
        mds_emb = rsv.vis.mds(rdms)
        self.assertEqual(mds_emb.shape, (8, 5, 2))


class Test_Icon(unittest.TestCase):

    def test_Icon_no_error(self):
        import PIL
        from pyrsa.vis import Icon
        import matplotlib.pyplot as plt
        test_im = PIL.Image.fromarray(255 * np.random.rand(50, 100))
        ic = Icon(image=255 * np.random.rand(50, 100), cmap='Blues')
        ax = plt.subplot(1, 1, 1)
        ic.plot(0.5, 0.5, ax=ax)
        ic2 = Icon(image=test_im, border_color='black', border_width=15,
                   string='test')
        ic2.plot(0.8, 0.2, ax=ax, size=0.4)
        ic2.x_tick_label(0.5, 0.15, offset=7)
        ic2.y_tick_label(0.5, 0.25, offset=7)
        ic3 = Icon(image=test_im, border_color='red', border_width=5,
                   make_square=True)
        ic3.plot(0.2, 0.2, size=0.4)
        ic4 = Icon(string='test')
        ic4.plot(0.2, 0.8, size=0.4)
        ic4.x_tick_label(0.75, 0.15, offset=7)
        ic4.y_tick_label(0.75, 0.25, offset=17)


if __name__ == '__main__':
    unittest.main()
