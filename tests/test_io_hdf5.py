from unittest import TestCase
from unittest.mock import patch
from importlib.metadata import version


class Hdf5IOTests(TestCase):

    @patch('rsatoolbox.util.file_io.File')
    def test_write_dict_hdf5_version(self, h5pyFile):
        """Check version tag matches current version
        """
        from rsatoolbox.util.file_io import write_dict_hdf5
        h5pyFile().attrs = dict()   
        write_dict_hdf5('a file path', dict())
        self.assertEqual(
            h5pyFile().attrs.get('rsatoolbox_version'),
            version('rsatoolbox')
        )
