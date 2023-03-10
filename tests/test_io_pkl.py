from unittest import TestCase
from unittest.mock import patch, sentinel
from importlib.metadata import version


class PickleIOTests(TestCase):

    @patch('rsatoolbox.io.pkl.pickle')
    def test_write_dict_pkl_version(self, pickle):
        """Check version tag matches current version
        """
        from rsatoolbox.io.pkl import write_dict_pkl
        write_dict_pkl(sentinel.filepath, dict())
        self.assertEqual(
            pickle.dump.call_args[0][0],
            dict(rsatoolbox_version=version('rsatoolbox')),
        )
