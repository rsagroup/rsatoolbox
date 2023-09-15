"""Test input/output for fmriprep 
"""
from unittest import TestCase
from unittest.mock import patch, Mock


class TestIoFmriprep(TestCase):

    @patch('rsatoolbox.io.fmriprep.BidsLayout')
    @patch('rsatoolbox.io.fmriprep.FmriprepRun')
    def test_find_fmriprep_runs(self, FmriprepRun, BidsLayout):
        from rsatoolbox.io.fmriprep import find_fmriprep_runs
        BidsLayout().find_derivative_files.return_value = ['a', 'b']
        FmriprepRun.side_effect = lambda f: 'run-'+f
        out = find_fmriprep_runs('/path')
        BidsLayout.assert_called_with('/path')
        self.assertEqual(out, ['run-a', 'run-b'])

    def test_FmriprepRun(self):
        from rsatoolbox.io.fmriprep import FmriprepRun
        bidsFile = Mock()
        run = FmriprepRun(bidsFile)
