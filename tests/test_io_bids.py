"""Tests BIDS Input/output
"""
from __future__ import annotations
from unittest import TestCase
from unittest.mock import patch, Mock


@patch('rsatoolbox.io.bids.BidsMriFile')
@patch('rsatoolbox.io.bids.glob')
@patch('rsatoolbox.io.bids.isdir')
class TestIoBids(TestCase):

    def test_BidsLayout_find_derivative_files(self, *patches):
        from rsatoolbox.io.bids import BidsLayout
        self.glob_will_return([
            '/root/derivatives/ana/sub-04/ses-03/mod/sub-04_ses-03_task-x_run-02_space-ikea_desc-t2.nii.gz',
            '/root/derivatives/ana/sub-04/ses-03/mod/sub-04_ses-03_task-x_run-02_space-ikea_desc-t2.json',
            '/root/derivatives/ana/sub-04/ses-03/mod/sub-04_ses-03_task-x_run-02_space-ikea_desc-t1.nii.gz',
            '/root/derivatives/ana/sub-04/ses-03/mod/sub-04_ses-03_task-x_run-02_space-ikea_desc-t1.json',
        ], *patches)
        layout = BidsLayout('/root', nibabel=self.nibabel)
        out = layout.find_mri_derivative_files(derivative='ana', desc='t1')
        self.assertEqual(out, [
            'derivatives/ana/sub-04/ses-03/mod/sub-04_ses-03_task-x_run-02_space-ikea_desc-t1.nii.gz',
        ])

    def test_BidsLayout_find_derivative_files_filters_tasks(self, *patches):
        from rsatoolbox.io.bids import BidsLayout
        self.glob_will_return([
            '/root/derivatives/ana/sub-04/ses-03/mod/sub-04_ses-03_task-x_run-02_space-ikea_desc-t1.nii.gz',
            '/root/derivatives/ana/sub-04/ses-03/mod/sub-04_ses-03_task-x_run-02_space-ikea_desc-t1.json',
            '/root/derivatives/ana/sub-04/ses-03/mod/sub-04_ses-03_task-y_run-02_space-ikea_desc-t1.nii.gz',
            '/root/derivatives/ana/sub-04/ses-03/mod/sub-04_ses-03_task-y_run-02_space-ikea_desc-t1.json',
        ], *patches)
        layout = BidsLayout('/root', nibabel=self.nibabel)
        out = layout.find_mri_derivative_files(derivative='ana', desc='t1', tasks=['y'])
        self.assertEqual(out, [
            'derivatives/ana/sub-04/ses-03/mod/sub-04_ses-03_task-y_run-02_space-ikea_desc-t1.nii.gz',
        ])
        
    def glob_will_return(self, fpaths, isdir, glob, BidsMriFile):
        """Setup the patches such that the passed file paths will be returned
        """
        isdir.return_value = True
        glob.return_value = fpaths
        BidsMriFile.side_effect = lambda f, b, n: f

    def setUp(self) -> None:
        self.nibabel = Mock()
        return super().setUp()
