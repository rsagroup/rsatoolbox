import unittest
import pkg_resources


class MeadowsIOTests(unittest.TestCase):
    """Acceptance and unit tests for loading Meadows data.
    """

    def test_load_dataset_from_mat_file(self):
        """Acceptance test for loading data from a Meadows
        .mat file download containing data for a single task,
        single participant. Should have descriptors and dissimilarities
        as found in file.
        """
        import pyrsa.io.meadows
        fname = 'Meadows_myExp_v_v1_cuddly-bunny_3_1D.mat'
        fpath = pkg_resources.resource_filename('tests', 'data/' + fname)
        myDataSet = pyrsa.io.meadows.load_datasets(fpath)
