#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for the simulation subpackage
"""

import unittest
import pyrsa.model as model
import numpy as np


class TestSimulation(unittest.TestCase):
    def test_make_design(self):
        import pyrsa.simulation.sim as sim
        # Test for make_design 
        cond_vec,part_vec = sim.make_design(4, 8)
        self.assertEqual(cond_vec.size,32)

    def test_make_data(self):
        # Test for make_data 
        import pyrsa.simulation.sim as sim
        cond_vec,part_vec = sim.make_design(4, 8)
        M = model.ModelFixed("test", np.array([2, 2, 2, 1, 1, 1]))
        D = sim.make_dataset(M, None, cond_vec, n_channel=40)
        self.assertEqual(D[0].n_obs, 32)
        self.assertEqual(D[0].n_channel, 40)


if __name__ == '__main__':
    unittest.main()
