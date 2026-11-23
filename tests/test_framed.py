"""
Tests for Framed RSA
"""
import unittest
import rsatoolbox.data as rsd
import numpy as np


class TestFramedDataset(unittest.TestCase):

    def setUp(self):
        self.rng = np.random.default_rng(0)
        self.n_stim = 10
        self.n_channel = 20
        self.n_fold = 5
        self.n_trials = self.n_stim * self.n_fold
        self.patterns_orig = self.rng.random((self.n_trials, self.n_channel))
        stim = np.repeat(np.arange(self.n_stim), self.n_fold)
        folds = np.tile(np.arange(self.n_fold), self.n_stim)
        self.dataset_basic = rsd.FramedDataset(measurements=self.patterns_orig,
                                               obs_descriptors={'stim': stim,
                                                                'fold': folds},
                                               cond_descriptor='stim',
                                               include_all_zeros=False,
                                               all_c_scale=None)
        self.dataset_zeros = rsd.FramedDataset(measurements=self.patterns_orig,
                                               obs_descriptors={'stim': stim,
                                                                'fold': folds},
                                               cond_descriptor='stim',
                                               include_all_zeros=True,
                                               all_c_scale=None)
        self.dataset_full = rsd.FramedDataset(measurements=self.patterns_orig,
                                              obs_descriptors={'stim': stim,
                                                               'fold': folds},
                                              cond_descriptor='stim',
                                              include_all_zeros=True,
                                              all_c_scale=1)

    def test_basic_shape(self):
        assert self.dataset_basic.measurements.shape[0] == self.n_trials

    def test_zeros_shape(self):
        assert self.dataset_zeros.measurements.shape[0] == self.n_trials + self.n_fold

    def test_full_shape(self):
        assert self.dataset_full.measurements.shape[0] == self.n_trials + self.n_fold * 2

    def test_sigmak(self):
        sigma_k_fromdata = self.dataset_full.get_sigmak(from_data=True, cv_desc='fold')
        sigma_k_notdata = self.dataset_full.get_sigmak(from_data=False, cv_desc='fold')
        assert sigma_k_fromdata.shape == (self.n_stim + 2, self.n_stim + 2)
        assert sigma_k_notdata.shape == (self.n_stim + 2, self.n_stim + 2)
        assert sigma_k_fromdata[0, 0] == 0
        assert sigma_k_fromdata[-1, -1] == 0
        assert sigma_k_notdata[0, 0] == 0
        assert sigma_k_notdata[-1, -1] == 0
        assert sigma_k_notdata[1, 1] == 1

    def test_mask(self):
        mask = self.dataset_full.get_framed_rdm_mask()
        assert mask.shape == (self.n_stim + 2, self.n_stim + 2)
        assert mask.sum() == 2 * (self.n_stim + 2) - 1
        assert mask[0, 0] == 1
        assert mask[-1, -1] == 1
        assert mask[1, 1] == 0
