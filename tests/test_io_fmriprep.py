"""Test input/output for fmriprep 
"""
from unittest import TestCase
from unittest.mock import patch, Mock


class TestIoFmriprep(TestCase):

    @patch('rsatoolbox.io.fmriprep.BidsLayout')
    @patch('rsatoolbox.io.fmriprep.FmriprepRun')
    def test_find_fmriprep_runs(self, FmriprepRun, BidsLayout):
        from rsatoolbox.io.fmriprep import find_fmriprep_runs
        BidsLayout().find_mri_derivative_files.return_value = ['a', 'b']
        FmriprepRun.side_effect = lambda f: 'run-'+f
        out = find_fmriprep_runs('/path')
        BidsLayout.assert_called_with('/path')
        self.assertEqual(out, ['run-a', 'run-b'])

    def test_FmriprepRun_siblings(self):
        from rsatoolbox.io.fmriprep import FmriprepRun
        bidsFile = Mock()
        brainMask = Mock()
        aparc = Mock()
        sibs = dict(
            brain_mask=brainMask,
            aparcaseg=aparc
        )
        bidsFile.get_mri_sibling.side_effect = lambda k: sibs[k]
        run = FmriprepRun(bidsFile)
        self.assertIs(run.get_brain_mask(), brainMask.get_data())
        self.assertIs(run.get_parcellation(), aparc.get_data())

    def test_FmriprepRun_to_descriptors(self):
        from rsatoolbox.io.fmriprep import FmriprepRun
        bidsFile = Mock()
        run = FmriprepRun(bidsFile)
        descs = run.to_descriptors()
