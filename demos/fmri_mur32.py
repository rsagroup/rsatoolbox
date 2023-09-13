"""
Work on a demo based on Marieke's 32-stim dataset

see also https://github.com/ilogue/mur32


## TODO

- [x] rename and move events files
- [x] copy data to worker
- [x] install fmriprep on worker
- [x] fmriprep 01-06
- [ ] fmriprep 07-14


roi as a bool descriptor?

"""

from os.path import expanduser, join
import json
import nibabel, pandas


data_dir = expanduser('~/data/rsatoolbox/mur32')
fpath = join(data_dir, 'sub-01', 'sub-01_run-01_events.tsv')
df = pandas.read_csv(fpath, sep='\t')

## pass optional descriptors == events columns
## ignore trial_types



## load_fmriprep(root_dir)
## load_fmriprep(root_dir, sub_index=3)

## io/bids
## io/fmriprep

