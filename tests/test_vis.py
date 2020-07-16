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


if __name__ == '__main__':
    unittest.main()
