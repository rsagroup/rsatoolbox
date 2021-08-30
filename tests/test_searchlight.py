"""
    searchlight tests
    @author: Daniel Lindh
"""
#pylint: disable=import-outside-toplevel, no-self-use
import unittest
import numpy as np


class TestSearchlight(unittest.TestCase):
    def test__get_searchlight_neighbors(self):
        from rsatoolbox.util.searchlight import _get_searchlight_neighbors

        mask = np.zeros((5, 5, 5))
        center = [2, 2, 2]
        mask[2, 2, 2] = 10
        radius = 2
        # a radius of 2 will give us
        neighbors = _get_searchlight_neighbors(mask, center, radius=radius)

        assert np.array(neighbors).shape  == (3, 27)
        assert np.mean(mask[tuple(neighbors)]) == 10/27

    def test_get_volume_searchlight(self):
        from rsatoolbox.util.searchlight import get_volume_searchlight

        mask = np.array([[[False, False, False],
                        [False,  True, False],
                        [False, False, False]],

                        [[False,  True, False],
                        [ True,  True,  True],
                        [False,  True, False]],

                        [[False, False, False],
                        [False,  True, False],
                        [False, False, False]]], dtype=int)


        centers, neighbors = get_volume_searchlight(mask, radius=1, threshold=1.0)
        assert len(centers) == 7
        assert len(neighbors) == 7

    def test_get_searchlight_RDMs(self):
        from rsatoolbox.util.searchlight import get_searchlight_RDMs

        n_observations = 5
        n_voxels = 5
        data_2d = np.random.random((n_observations, n_voxels))
        centers = np.array([1, 3])
        neighbors = [[0,1,2], [2,3,4]]
        events = np.arange(n_observations)

        sl_RDMs = get_searchlight_RDMs(data_2d, centers, neighbors, events)

        assert sl_RDMs.dissimilarities.shape == (2, 10)
