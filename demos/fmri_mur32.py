"""
Work on a demo based on Marieke's 32-stim dataset

see also ilogue/mur32
"""

from os.path import expanduser, join
import json
import nibabel, pandas


data_dir = expanduser('~/data/rsatoolbox/mur32')
fpath = join(data_dir, 'sub-01', 'sub-01_run-01_events.tsv')
df = pandas.read_csv(fpath, sep='\t')

## pass optional descriptors == events columns
## ignore trial_types

