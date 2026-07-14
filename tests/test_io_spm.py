"""Tests for SPM I/O functions
"""
from __future__ import annotations
from typing import Dict
from unittest import TestCase
from unittest.mock import Mock, patch
import numpy as np
from os.path import join


class TestIoSPM(TestCase):

    def setUp(self) -> None:
        self.nitools = Mock()

    def stub_spm_mat(self) -> Dict:
        return {'SPM':
            {
                'nscan': [1, 2, 3],
                'Vbeta': [dict(fname='a')],
                'xY': {
                    'P': [
                        '/Users/jdoe/DoeLab Dropbox/the_proj/func/uas01_run01.nii,1  ',
                    ],
                },
                'xX': {
                    'name': ['00012 b'],
                    'K': [dict(X0=None)],
                    'iC': np.array([1]),
                    'xKXs': dict(X=None),
                    'erdf': None,
                    'W': None,
                    'pKX': None
                }
            }
        }

    @patch('rsatoolbox.io.spm.loadmat')
    def test_basic_spmglm_usage(self, loadmat):
        loadmat.return_value = self.stub_spm_mat()
        self.nitools.sample_images.return_value = np.array([[4, 5, 6]])
        from rsatoolbox.io.spm import SpmGlm
        spm = SpmGlm('/path', self.nitools)
        spm.get_info_from_spm_mat()
        [beta, _, info] = spm.get_betas('/pth/anat/M1_L.nii')
        self.assertEqual(beta.shape, (0, 3))
        self.assertEqual(info['reg_name'][0], 'b')
        self.assertEqual(info['run_number'][0], 1)

    @patch('rsatoolbox.io.spm.loadmat')
    def test_adapt_spm_paths(self, loadmat):
        loadmat.return_value = self.stub_spm_mat()
        from rsatoolbox.io.spm import SpmGlm
        spm = SpmGlm('/path/glm_firstlevel', self.nitools)
        spm.get_info_from_spm_mat()
        self.assertEqual(spm.rawdata_files, [
            '/path/func/uas01_run01.nii,1  ',
        ])

    def test_relocate_file(self):
        from rsatoolbox.io.spm import SpmGlm
        spm = SpmGlm(join('/path', 'leaf'), self.nitools)
        self.assertEqual(
            spm.relocate_file('/bla/dip/func/abc.nii,1  '),
            '/path/func/abc.nii,1  '
        )
        self.assertEqual(
            spm.relocate_file('c:\\bla\\dip\\func\\abc.nii,2  '),
            '/path/func/abc.nii,2  '
        )
