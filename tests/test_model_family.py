"""
test_colors
Test for colors
@author: iancharest
"""

import unittest
import numpy as np
from pyrsa.model.model_family import  ModelFamily
from pyrsa.vis.colors import rdm_colormap


class ModelFamilyTest(unittest.TestCase):

    def test_family_creation(self):
        random_list = [4,6,5,2]
        model_family = ModelFamily(random_list)
        print(model_family.family_list)
        print(model_family.model_indices)

if __name__ == '__main__':
    unittest.main()
