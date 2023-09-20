"""Test input/output for fmriprep 
"""
from unittest import TestCase
from unittest.mock import patch, Mock
from numpy.testing import assert_array_equal
import pandas, numpy


class TestFindFmriprepRuns(TestCase):

    @patch('rsatoolbox.io.fmriprep.BidsLayout')
    @patch('rsatoolbox.io.fmriprep.FmriprepRun')
    def test_find_fmriprep_runs(self, FmriprepRun, BidsLayout):
        from rsatoolbox.io.fmriprep import find_fmriprep_runs
        BidsLayout().find_mri_derivative_files.return_value = ['a', 'b']
        FmriprepRun.side_effect = lambda f: 'run-'+f
        out = find_fmriprep_runs('/path')
        BidsLayout.assert_called_with('/path')
        self.assertEqual(out, ['run-a', 'run-b'])


class TestFmriprepRun(TestCase):

    def test_FmriprepRun_siblings(self):
        from rsatoolbox.io.fmriprep import FmriprepRun
        bidsFile = Mock()
        mask = Mock()
        parc = Mock()
        sibs = dict(
            brain=mask,
            aparcaseg=parc
        )
        bidsFile.get_mri_sibling.side_effect = lambda desc, **kw: sibs[desc]
        run = FmriprepRun(bidsFile)
        self.assertIs(run.get_mask(), mask.get_data().astype(bool))
        self.assertIs(run.get_parcellation(), parc.get_data().astype(int))

    def test_FmriprepRun_dataset_descriptors(self):
        from rsatoolbox.io.fmriprep import FmriprepRun
        bidsFile = Mock()
        bidsFile.modality = 'moda'
        bidsFile.sub = '05'
        bidsFile.ses = '04'
        bidsFile.task = 'T1'
        bidsFile.run = '03'
        bidsFile.mod = 'mod'
        run = FmriprepRun(bidsFile)
        descs = run.get_dataset_descriptors()
        self.assertEqual(descs, dict(
            sub='05', ses='04', run='03', task='T1'
        ))

    def test_FmriprepRun_obs_descriptors(self):
        from rsatoolbox.io.fmriprep import FmriprepRun
        bidsFile = Mock()
        bidsFile.get_events.return_value = pandas.DataFrame([
            dict(trial_type='s1'),
            dict(trial_type='s2'),
            dict(trial_type='s1'),
            dict(trial_type='s3'),
        ])
        run = FmriprepRun(bidsFile)
        descs = run.get_obs_descriptors()
        self.assertIn('trial_type', descs)
        self.assertEqual(
            list(descs['trial_type']),
            ['s1', 's2', 's1', 's3']
        )

    def test_FmriprepRun_obs_descriptors_collapsed(self):
        """If we set collapse_by_trial_type=true,
        observations should be collapsed by trial_type.
        """
        from rsatoolbox.io.fmriprep import FmriprepRun
        bidsFile = Mock()
        bidsFile.get_events.return_value = pandas.DataFrame([
            dict(trial_type='s1'),
            dict(trial_type='s2'),
            dict(trial_type='s1'),
            dict(trial_type='s3'),
        ])
        run = FmriprepRun(bidsFile)
        descs = run.get_obs_descriptors(collapse_by_trial_type=True)
        self.assertIn('trial_type', descs)
        self.assertEqual(
            list(descs['trial_type']), 
            ['s1', 's2', 's3']
        )

    def test_FmriprepRun_channel_descriptors(self):
        from rsatoolbox.io.fmriprep import FmriprepRun
        bidsFile = Mock()
        parc = Mock()
        parc.get_data.return_value = numpy.array([[0, 2], [0, 4]])
        df = pandas.DataFrame([
            dict(index=0, name='nothing'),
            dict(index=2, name='foo'),
            dict(index=4, name='bar'),
        ])
        parc.get_key().get_frame.return_value = df
        bidsFile.get_mri_sibling.return_value = parc
        run = FmriprepRun(bidsFile)
        descs = run.get_channel_descriptors()
        self.assertIn('aparcaseg', descs)
        assert_array_equal(
            descs['aparcaseg'],
            ['nothing', 'foo', 'nothing', 'bar']
        )


class TestEventsDesignMatrix(TestCase):

    def test_make_design_matrix(self):
        from rsatoolbox.io.fmriprep import make_design_matrix
        events = pandas.DataFrame([
            dict(onset=1, duration=0.5, trial_type='a'),
            dict(onset=2, duration=0.5, trial_type='b'),
            dict(onset=3, duration=0.5, trial_type='a'),
            dict(onset=4, duration=0.5, trial_type='b'),
            dict(onset=5, duration=0.5, trial_type='a'),
            dict(onset=6, duration=0.5, trial_type='b'),
        ])
        confounds = pandas.DataFrame([
            dict(sig1=1.0, sig2=0.5),
            dict(sig1=1.1, sig2=0.4),
            dict(sig1=1.2, sig2=0.3),
            dict(sig1=1.3, sig2=0.2),
        ])
        dm, pred_mask, dof = make_design_matrix(events, tr=2.0, n_vols=4,
                                           confounds=confounds)
        self.assertEqual(dm.shape, (4, 2+2))
        self.assertEqual(dof, 0)
        assert_array_equal(pred_mask, [True, True, False, False])
