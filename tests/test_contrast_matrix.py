import unittest
import numpy as np

import pyrsa.util.contrast_matrix as contrast_matrix

class TestContrastMatrix(unittest.TestCase):
    
    def test_contrast_matrix(self):
        C = contrast_matrix(3)
        assert C.shape == (3,3)
        assert np.all(C==np.array([[1,-1,0],[1,0,-1],[0,1,-1]]))
