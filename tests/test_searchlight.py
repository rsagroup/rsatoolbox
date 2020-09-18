    """
    

    """


import unittest
import numpy as np
from parameterized import parameterized


class TestSearchlight(unittest.TestCase):
    def test__get_searchlight_neighbors(self):
        from pyrsa.util.searchlight import _get_searchlight_neighbors

        mask = np.zeros((5, 5, 5))
        center = [2, 2, 2]
        mask[2, 2, 2] = 10
        radius = 3
        # a radius of 2 will give us
        neighbors = _get_searchlight_neighbors(mask, center, radius=radius)

        assert np.array(neighbors).shape  == (3, 27)

