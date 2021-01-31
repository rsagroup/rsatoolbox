#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 17:23:33 2020

@author: heiko
"""

import stats
import numpy as np

for i in np.random.permutation(14400):
    stats.run_comp(i)
