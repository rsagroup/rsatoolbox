#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 17:54:33 2020

@author: heiko
"""

import os
import pathlib
import numpy as np

for folder in ['comp_zero', 'comp_model', 'comp_noise']:
    for p in pathlib.Path(folder).glob('p_*'):
        split = p.name.split('_')
        if split[2] == 'boot':
            split[2] = 'both'
        if split[3] in  ['t', 'fix', 'perc', 'ranksum']:
            pass
        else:
            print('')
            print(os.path.join(folder, p.name))
            split = split[:3] + ['perc'] + split[3:]
            new_name = '_'.join(split)
            print(os.path.join(folder, new_name))
            os.rename(os.path.join(folder, p.name),
                      os.path.join(folder, new_name))


for folder in ['comp_zero', 'comp_model', 'comp_noise']:
    for p in pathlib.Path(folder).glob('p_*'):
        split = p.name.split('_')
        if split[3] == 'perc':
            print(os.path.join(folder, p.name))
            dat = np.load(os.path.join(folder, p.name))
            if folder == 'comp_zero':
                dat = 1 - dat
            else:
                dat = 2 * np.minimum(dat, 1 - dat)
            np.save(os.path.join(folder, p.name), dat)
