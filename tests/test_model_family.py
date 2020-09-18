"""
test_colors
Test for colors
@author: iancharest
"""

import unittest
import numpy as np
from pyrsa.model import  ModelFamily
from pyrsa.vis.colors import rdm_colormap


class ModelFamilyTest(unittest.TestCase):

    def test_family_creation(self):
        random_list = [4,6,5,2,99]
        model_family = ModelFamily(random_list)
        
        print("-----------------Model family tree( indices)------------------")
        for i in range(model_family.num_family_members):
            print(model_family.family_list[i])
        
        print("-----------------Model family binary indices------------------")
        print(model_family.model_indices)
        
        print("-----------------Model family tree(members)------------------")
        for i in range(model_family.num_family_members):
            print(model_family.get_family_member(i))

if __name__ == '__main__':
    unittest.main()
