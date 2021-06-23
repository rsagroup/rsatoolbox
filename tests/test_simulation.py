#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for the simulation subpackage
"""
#pylint: disable=import-outside-toplevel, no-self-use
import unittest
import numpy as np
from scipy.spatial.distance import squareform
import rsatoolbox
import rsatoolbox.model as model


class TestSimulation(unittest.TestCase):

    def test_make_design(self):
        import rsatoolbox.simulation.sim as sim
        # Test for make_design
        cond_vec, _ = sim.make_design(4, 8)
        self.assertEqual(cond_vec.size, 32)

    def test_make_signal(self):
        # Test make signal
        import rsatoolbox.simulation.sim as sim
        M = model.ModelFixed("test", np.array([2, 2, 2, 1, 1, 1]))
        RDM = M.predict(None)
        D = squareform(RDM)
        H = rsatoolbox.util.matrix.centering(D.shape[0])
        G = -0.5 * (H @ D @ H)
        S = sim.make_signal(G, 40, make_exact=True)
        Diff = S@S.T/40 - G
        self.assertTrue(np.all(np.abs(Diff) < 1e-7))

    def test_make_data(self):
        # Test for make_data
        import rsatoolbox.simulation.sim as sim
        cond_vec, _ = sim.make_design(4, 8)
        M = model.ModelFixed("test", np.array([2, 3, 4, 1, 1.1, 0.9]))
        D = sim.make_dataset(M, None, cond_vec, n_channel=40)
        self.assertEqual(D[0].n_obs, 32)
        self.assertEqual(D[0].n_channel, 40)


if __name__ == '__main__':
    unittest.main()
