"""Tests for SPM I/O functions
"""
from unittest import TestCase
from unittest.mock import Mock
import numpy as np


class TestIoSPM(TestCase):

    def setUp(self) -> None:
        self.nibabel = Mock()
        img = Mock()
        img.get_fdata.return_value = np.zeros([3, 4, 5])
        self.nibabel.nifti1.load.return_value = img
        self.glob = Mock()
        self.glob.return_value = ['a', 'b']

    def test_beta_pooling(self):
        from rsatoolbox.io.spm import SpmGlm
        glm = SpmGlm('/path', self.nibabel, self.glob)
        glm.load_betas()
        np.testing.assert_array_equal(
            glm.pooled_data_array.shape, (3, 4, 5, 2))
        np.testing.assert_array_equal(
            glm.pooled_data_array.shape[3], len(glm.dim4_descriptors))
        glm.dim4_descriptors.sort()
        np.testing.assert_array_equal(
            np.unique(glm.dim4_descriptors),
            glm.dim4_descriptors)

    def test_beta_pooling_w_dict(self):
        from rsatoolbox.io.spm import SpmGlm
        glm = SpmGlm('/path', self.nibabel, self.glob)
        glm.load_betas()
        np.testing.assert_array_equal(
            glm.pooled_data_array.shape, (3, 4, 5, 2))
        np.testing.assert_array_equal(
            glm.pooled_data_array.shape[3], len(glm.dim4_descriptors))
        glm.dim4_descriptors.sort()
        np.testing.assert_array_equal(
            np.unique(glm.dim4_descriptors),
            glm.dim4_descriptors)

    def test_res_pooling(self):
        from rsatoolbox.io.spm import SpmGlm
        glm = SpmGlm('/path', self.nibabel, self.glob)
        glm.load_residuals()
        np.testing.assert_array_equal(
            glm.pooled_data_array.shape, (3, 4, 5, 2))
        np.testing.assert_array_equal(
            glm.pooled_data_array.shape[3], len(glm.dim4_descriptors))
        glm.dim4_descriptors.sort()
        np.testing.assert_array_equal(
            np.unique(glm.dim4_descriptors),
            glm.dim4_descriptors)
