from unittest import TestCase
import pkg_resources
from numpy.testing import assert_array_equal, assert_array_almost_equal


class MeadowsIOTests(TestCase):
    """Acceptance and unit tests for loading Meadows data.
    """

    def test_load_rdms_from_mat_file_1p(self):
        """Acceptance test for loading data from a Meadows
        .mat file download containing data for a single task,
        single participant. Should have descriptors and dissimilarities
        as found in file.
        """
        import rsatoolbox.io.meadows
        fname = 'Meadows_myExp_v_v1_cuddly-bunny_3_1D.mat'
        fpath = pkg_resources.resource_filename('tests', 'data/' + fname)
        rdms = rsatoolbox.io.meadows.load_rdms(fpath, sort=False)
        self.assertEqual(rdms.descriptors.get('participant'), 'cuddly-bunny')
        self.assertEqual(rdms.descriptors.get('task_index'), 3)
        self.assertEqual(rdms.descriptors.get('experiment_name'), 'myExp')
        self.assertEqual(rdms.dissimilarity_measure, 'euclidean')
        conds = rdms.pattern_descriptors.get('conds')
        assert_array_equal(conds[:2], ['stim118', 'stim117'])
        assert_array_almost_equal(
            rdms.dissimilarities[0, :2],
            [0.00791285387561264, 0.00817090931233484]
        )

    def test_load_rdms_and_sort(self):
        """As above but with sorting
        """
        import rsatoolbox.io.meadows
        fname = 'Meadows_myExp_v_v1_cuddly-bunny_3_1D.mat'
        fpath = pkg_resources.resource_filename('tests', 'data/' + fname)
        rdms = rsatoolbox.io.meadows.load_rdms(fpath, sort=True)
        conds = rdms.pattern_descriptors.get('conds')
        assert_array_equal(conds[:2], ['stim001', 'stim002'])
        assert_array_equal(conds[-2:], ['stim117', 'stim118'])
        assert_array_almost_equal(
            rdms.dissimilarities[0, -2:],
            [0.00817090931233484, 0.00791285387561264]
        )

    def test_load_rdms_from_mat_file_3p(self):
        """Acceptance test for loading data from a Meadows
        .mat file download containing data for a single task,
        multiple participants. Should have descriptors and dissimilarities
        as found in file.
        """
        import rsatoolbox.io.meadows
        fname = 'Meadows_myExp_v_v1_arrangement_1D.mat'
        fpath = pkg_resources.resource_filename('tests', 'data/' + fname)
        rdms = rsatoolbox.io.meadows.load_rdms(fpath, sort=False)
        self.assertEqual(rdms.descriptors.get('task_name'), 'arrangement')
        self.assertEqual(rdms.descriptors.get('experiment_name'), 'myExp')
        self.assertEqual(
            rdms.rdm_descriptors.get('participants'),
            ['able-fly', 'clean-koi', 'cuddly-bunny']
        )
        self.assertEqual(rdms.dissimilarity_measure, 'euclidean')
        conds = rdms.pattern_descriptors.get('conds')
        assert_array_equal(conds[:2], ['stim118', 'stim117'])
        assert_array_almost_equal(
            rdms.dissimilarities[0, :2],  # 'able-fly'
            [0.0165981067918494, 0.0123233998529090]
        )
        assert_array_almost_equal(
            rdms.dissimilarities[1, :2],  # 'clean-koy'
            [0.00773234353884765, 0.00589909056106329]
        )

    def test_extract_filename_segments_1p_1t(self):
        """Test interpretation of the filename of a Meadows results download

        This case covers a filename for a single participant, single task,
        downloaded in 2019.
        """
        from rsatoolbox.io.meadows import extract_filename_segments
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
        self.assertIsNone(info.get('task_name'))

    def test_extract_filename_segments_mp_1t(self):
        """Test interpretation of the filename of a Meadows results download

        This case covers a filename for multiple participants, single task,
        downloaded in 2019.
        """
        from rsatoolbox.io.meadows import extract_filename_segments
        fname = 'Meadows_myExp_v_v1_arrangement_1D.mat'
        info = extract_filename_segments(fname)
        self.assertEqual(info.get('participant_scope'), 'multiple')
        self.assertEqual(info.get('task_scope'), 'single')
        self.assertEqual(info.get('task_name'), 'arrangement')
        self.assertEqual(info.get('version'), '1')
        self.assertEqual(info.get('experiment_name'), 'myExp')
        self.assertEqual(info.get('structure'), '1D')
        self.assertEqual(info.get('filetype'), 'mat')
        self.assertIsNone(info.get('participant'))
        self.assertIsNone(info.get('task_index'))
