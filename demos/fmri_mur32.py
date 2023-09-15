"""
Work on a demo based on Marieke's 32-stim dataset

see also https://github.com/ilogue/mur32


## TODO

- [x] rename and move events files
- [x] copy data to worker
- [x] install fmriprep on worker
- [x] fmriprep 01-06
- [ ] fmriprep 07-14
- [ ] roi as a bool descriptor?

"""
from os.path import expanduser
from rsatoolbox.io.fmriprep import find_fmriprep_runs
from rsatoolbox.data.dataset import Dataset


data_dir = expanduser('~/data/rsatoolbox/mur32')

runs = find_fmriprep_runs(data_dir, tasks=['main'])


## FIRST DO THIS STEP BY STEP FOR ONE ENTRY, then loop
datasets = []
for run in find_fmriprep_runs(data_dir, tasks=['main']):
    patterns = simple_glm(run)
    ds = Dataset(
        measurements=patterns,
        **run.to_descriptors()
    )
    datasets.append(ds)


## display structure of DS here

## display glass brain atlas regions (wishlist)

## Create RDM per subject and region
# - use runs for noise


## Inference on subjects


