"""
    searchlight tests
    @author: Daniel Lindh
"""
# pylint: disable=import-outside-toplevel, no-self-use
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

        assert np.array(neighbors).shape == (3, 27)
        assert np.mean(mask[tuple(neighbors)]) == 10 / 27

    def test_get_volume_searchlight(self):
        from rsatoolbox.util.searchlight import get_volume_searchlight

        mask = np.array(
            [[[False, False, False],
              [False, True, False],
              [False, False, False]],

             [[False, True, False],
             [True, True, True],
             [False, True, False]],

             [[False, False, False],
             [False, True, False],
             [False, False, False]]], dtype=int)

        centers, neighbors = get_volume_searchlight(
            mask, radius=1, threshold=1.0)
        assert len(centers) == 7
        assert len(neighbors) == 7

    def test_get_searchlight_RDMs(self):
        from rsatoolbox.util.searchlight import get_searchlight_RDMs

        n_observations = 5
        n_voxels = 5
        rng = np.random.default_rng(0)
        data_2d = rng.random((n_observations, n_voxels))
        centers = np.array([1, 3])
        neighbors = [[0, 1, 2], [2, 3, 4]]
        events = np.arange(n_observations)

        sl_RDMs = get_searchlight_RDMs(data_2d, centers, neighbors, events)

        assert sl_RDMs.dissimilarities.shape == (2, 10)

    def test_boundary_truncation_disabled(self):
        """Test that truncate_at_boundary=False includes voxels outside mask (default behavior)"""
        from rsatoolbox.util.searchlight import _get_searchlight_neighbors

        # Create a mask where only center voxel is True
        mask = np.zeros((5, 5, 5), dtype=bool)
        mask[2, 2, 2] = True
        center = [2, 2, 2]
        radius = 2

        # With truncation disabled, should include all voxels within radius
        neighbors = _get_searchlight_neighbors(mask, center, radius=radius, truncate_at_boundary=False)

        # A radius of 2 should give 27 voxels (including those outside mask)
        assert np.array(neighbors).shape == (3, 27)
        # Only 1 of the 27 voxels is in the mask
        assert np.sum(mask[neighbors]) == 1

    def test_boundary_truncation_enabled(self):
        """Test that truncate_at_boundary=True filters out voxels outside mask"""
        from rsatoolbox.util.searchlight import _get_searchlight_neighbors

        # Create a mask where only center voxel is True
        mask = np.zeros((5, 5, 5), dtype=bool)
        mask[2, 2, 2] = True
        center = [2, 2, 2]
        radius = 2

        # With truncation enabled, should only include voxels where mask is True
        neighbors = _get_searchlight_neighbors(mask, center, radius=radius, truncate_at_boundary=True)

        # Should only include the 1 voxel where mask is True
        assert np.array(neighbors).shape == (3, 1)
        # All returned voxels should be in the mask
        assert np.all(mask[neighbors])

    def test_boundary_truncation_partial_mask(self):
        """Test truncation with a partial mask (boundary case)"""
        from rsatoolbox.util.searchlight import _get_searchlight_neighbors

        # Create a mask with a cross pattern
        mask = np.zeros((5, 5, 5), dtype=bool)
        mask[2, 2, :] = True  # line along z axis
        mask[2, :, 2] = True  # line along y axis
        mask[:, 2, 2] = True  # line along x axis
        center = [2, 2, 2]
        radius = 2

        # Without truncation
        neighbors_no_trunc = _get_searchlight_neighbors(mask, center, radius=radius, truncate_at_boundary=False)
        n_voxels_no_trunc = np.array(neighbors_no_trunc).shape[1]

        # With truncation
        neighbors_trunc = _get_searchlight_neighbors(mask, center, radius=radius, truncate_at_boundary=True)
        n_voxels_trunc = np.array(neighbors_trunc).shape[1]

        # Truncated should have fewer voxels
        assert n_voxels_trunc < n_voxels_no_trunc
        # All truncated voxels should be in the mask
        assert np.all(mask[neighbors_trunc])
        # Not all non-truncated voxels should be in the mask
        assert not np.all(mask[neighbors_no_trunc])

    def test_get_volume_searchlight_with_truncation(self):
        """Test that get_volume_searchlight passes truncation parameter correctly"""
        from rsatoolbox.util.searchlight import get_volume_searchlight

        # Create a simple mask
        mask = np.zeros((5, 5, 5), dtype=int)
        mask[1:4, 1:4, 1:4] = 1  # 3x3x3 cube in the center

        # Test with truncation disabled (default)
        centers_no_trunc, _ = get_volume_searchlight(
            mask, radius=1, threshold=0.5, truncate_at_boundary=False)

        # Test with truncation enabled
        centers_trunc, neighbors_trunc = get_volume_searchlight(
            mask, radius=1, threshold=0.5, truncate_at_boundary=True)

        # Should find centers in both cases
        assert len(centers_no_trunc) > 0
        assert len(centers_trunc) > 0

        # With truncation, all neighbor voxels should be in the mask
        for neighbors in neighbors_trunc:
            neighbor_coords = np.unravel_index(neighbors, mask.shape)
            assert np.all(mask[neighbor_coords] == 1)
