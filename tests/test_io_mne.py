from unittest import TestCase
from os.path import join
from tempfile import TemporaryDirectory
from numpy.testing import assert_almost_equal
import numpy


class MneIOTests(TestCase):
    """Acceptance and unit tests for loading MNE data.
    """

    def setUp(self) -> None:
        """Assemble MNE Epochs object from toy data
        """
        import mne  # should be installed as a test dependency
        self.test_dir = TemporaryDirectory()
        self.events = numpy.array([
            [200, 0, 11],
            [1200, 0, 12],
            [2000, 0, 13],
            [2200, 0, 12],
        ])
        t = numpy.arange(12).reshape([4, 3])
        data = numpy.array([numpy.sin(t), numpy.cos(t)])
        self.data = numpy.moveaxis(data, 0, 1)
        info = mne.create_info(
            ch_names=['A1', 'X32'],
            ch_types='eeg',
            sfreq=20
        )
        self.epochs = mne.EpochsArray(self.data, info, self.events)
        return super().setUp()

    def store_test_epochs(self, fname: str) -> str:
        """Save the epochs object with this filename and return the full path
        """
        fpath = join(self.test_dir.name, fname)
        self.epochs.save(fpath, verbose='error')
        return fpath

    def tearDown(self) -> None:
        """Delete any files created
        """
        self.test_dir.cleanup()
        return super().tearDown()

    def test_load_epochs(self):
        """Acceptance test for loading a single MNE _epo.fif
        file as TemporalDataset
        """
        from rsatoolbox.io.mne import load_epochs
        ds = load_epochs(self.store_test_epochs('test_epo.fif'))
        self.assertEqual(ds.measurements.shape, (4, 2, 3))
        assert_almost_equal(ds.measurements, self.data)
        self.assertEqual(ds.descriptors.get('filename'), 'test_epo.fif')
        # events desc
        self.assertEqual(
            ds.channel_descriptors.get('name'),
            ['A1', 'X32']
        )
        assert_almost_equal(
            ds.time_descriptors.get('time', []),
            [0, 0.05, 0.1]
        )

    def test_load_descriptors_from_bids_filename(self):
        """read BIDS-style tab-separated events file as additional
        obs-descriptor
        """
        from rsatoolbox.io.mne import load_epochs
        fpath = self.store_test_epochs('sub-01_run-02_task-abc_epo.fif')
        ds = load_epochs(fpath)
        self.assertEqual(ds.descriptors.get('sub'), '01')
        self.assertEqual(ds.descriptors.get('run'), '02')
        self.assertEqual(ds.descriptors.get('task'), 'abc')
