import unittest
import pkg_resources


class MeadowsIOTests(unittest.TestCase):
    """Acceptance and unit tests for loading Meadows data.
    """

    def test_load_rdms_from_mat_file(self):
        """Acceptance test for loading data from a Meadows
        .mat file download containing data for a single task,
        single participant. Should have descriptors and dissimilarities
        as found in file.
        """
        import pyrsa.io.meadows
        fname = 'Meadows_myExp_v_v1_cuddly-bunny_3_1D.mat'
        fpath = pkg_resources.resource_filename('tests', 'data/' + fname)
        rdms = pyrsa.io.meadows.load_rdms(fpath)
        self.assertEqual(rdms.descriptors.get('participant'), 'cuddly-bunny')
        self.assertEqual(rdms.descriptors.get('task_index'), 3)
        self.assertEqual(rdms.descriptors.get('experiment_name'), 'myExp')
        self.assertEqual(rdms.dissimilarity_measure, 'euclidean')
        conds = rdms.pattern_descriptors.get('conds')
        self.assertEqual(conds[:1], ['stim118', 'stim117'])
        self.assertEqual(
            rdms.dissimilarities[:2],
            [0.00791285387561264, 0.00817090931233484]
        )

    def test_extract_filename_segments_1p_1t(self):
        """Test interpretation of the filename of a Meadows results download

        This case covers a filename for a single participant, single task,
        downloaded in 2019.
        """
        from pyrsa.io.meadows import extract_filename_segments
        fname = 'Meadows_myExp_v_v1_cuddly-bunny_3_1D.mat'
        info = extract_filename_segments(fname)
        self.assertEqual(info.get('participant_scope'), 'single')
        self.assertEqual(info.get('task_scope'), 'single')
        self.assertEqual(info.get('participant'), 'cuddly-bunny')
        self.assertEqual(info.get('task_index'), 3)
        self.assertEqual(info.get('version'), '1')
        self.assertEqual(info.get('experiment_name'), 'myExp')
        self.assertEqual(info.get('structure'), '1D')
        self.assertEqual(info.get('filetype'), 'mat')
