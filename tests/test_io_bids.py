"""Tests BIDS Input/output
"""
from __future__ import annotations
from typing import List
from unittest import TestCase
from unittest.mock import patch, Mock
import os


@patch('rsatoolbox.io.bids.BidsTableFile')
@patch('rsatoolbox.io.bids.BidsMriFile')
@patch('rsatoolbox.io.bids.glob')
@patch('rsatoolbox.io.bids.isdir')
class TestBidsLayout(TestCase):

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
        self.assertEqual(out, self.os_paths_like([
            'derivatives/ana/sub-04/ses-03/mod/sub-04_ses-03_task-x_run-02_space-ikea_desc-t1.nii.gz',
        ]))

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
        self.assertEqual(out, self.os_paths_like([
            'derivatives/ana/sub-04/ses-03/mod/sub-04_ses-03_task-y_run-02_space-ikea_desc-t1.nii.gz',
        ]))

    def test_find_table_key_for(self, *patches):
        from rsatoolbox.io.bids import BidsLayout
        self.glob_will_return([], *patches)
        layout = BidsLayout('/root', nibabel=self.nibabel)
        derivFile = Mock()
        derivFile = Mock()
        derivFile.derivative = 'abra'
        derivFile.modality = 'moda'
        derivFile.desc = 'bla'
        derivFile.sub = '05'
        derivFile.space = None
        derivFile.suffix = 'mod'
        derivFile.ext = 'foo.bz'
        out = layout.find_table_key_for(derivFile)
        self.assertEqual(out, self.os_path_like(
            'derivatives/abra/desc-bla_mod.tsv'
        ))

    def test_BidsLayout_find_events_for(self, *patches):
        from rsatoolbox.io.bids import BidsLayout
        self.glob_will_return([], *patches)
        layout = BidsLayout('/root', nibabel=self.nibabel)
        derivFile = Mock()
        derivFile.modality = 'moda'
        derivFile.sub = '05'
        derivFile.ses = '04'
        derivFile.task = 'T1'
        derivFile.run = '03'
        derivFile.mod = 'mod'
        out = layout.find_events_for(derivFile)
        self.assertEqual(out, self.os_path_like(
            'sub-05/ses-04/moda/sub-05_ses-04_task-T1_run-03_events.tsv'
        ))
        
    def test_BidsLayout_find_mri_sibling_of(self, *patches):
        from rsatoolbox.io.bids import BidsLayout
        self.glob_will_return([], *patches)
        layout = BidsLayout('/root', nibabel=self.nibabel)
        derivFile = Mock()
        derivFile.derivative = 'abra'
        derivFile.modality = 'moda'
        derivFile.sub = '05'
        derivFile.ses = '04'
        derivFile.space = None
        derivFile.task = 'T1'
        derivFile.run = '03'
        derivFile.suffix = 'mod'
        derivFile.ext = 'foo.bz'
        out = layout.find_mri_sibling_of(derivFile, desc='bla', suffix='mod')
        self.assertEqual(out, self.os_path_like(
            'derivatives/abra/sub-05/ses-04/moda/sub-05_ses-04_task-T1_run-03_desc-bla_mod.foo.bz'
        ))

    def glob_will_return(self, fpaths, isdir, glob, BidsMriFile, BidsTableFile):
        """Setup the patches such that the passed file paths will be returned
        """
        isdir.return_value = True
        glob.return_value = fpaths
        BidsMriFile.side_effect = lambda f, b, n: f
        BidsTableFile.side_effect = lambda f, b: f

    def os_path_like(self, nix_path: str) -> str:
        return nix_path.replace('/', os.sep)
    
    def os_paths_like(self, nix_paths: List[str]) -> List[str]:
        return [self.os_path_like(p) for p in nix_paths]

    def setUp(self) -> None:
        self.nibabel = Mock()
        return super().setUp()


class TestBidsFile(TestCase):

    def test_deconstruct_entities(self):
        from rsatoolbox.io.bids import BidsFile
        layout = Mock()
        file = BidsFile('derivatives/ana/sub-04/ses-03/mod/' +
            'sub-04_ses-03_task-Tx_run-02_space-ikea_desc-t1_bar.nii.gz', layout)
        self.assertEqual(file.derivative, 'ana')
        self.assertEqual(file.sub, '04')
        self.assertEqual(file.ses, '03')
        self.assertEqual(file.modality, 'mod')
        self.assertEqual(file.task, 'Tx')
        self.assertEqual(file.run, '02')
        self.assertEqual(file.suffix, 'bar')
        self.assertEqual(file.ext, 'nii.gz')
