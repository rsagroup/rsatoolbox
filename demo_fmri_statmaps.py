"""Script version of statmap-based fmri demo

This approach: store full statmaps
Other approach: store only masked roi patterns

"""
from os.path import expanduser, join
import json, warnings
import numpy
from nibabel.nifti1 import Nifti1Image
from rsatoolbox.data.dataset import Dataset


data_dir = expanduser('~/data/rsatoolbox/mur32/derivatives/nilearn')

print('Loading..')
with open(join(data_dir, 'meta.json')) as fhandle:
    metadata = json.load(fhandle)
data = numpy.load(join(data_dir, 'data.npz'))
rois, betas = data['rois'], data['betas']
subjects = sorted(set(metadata['subjects']))
conditions = metadata['conditions']
N_RUNS = 6


for roi, region_name in enumerate(metadata['region_names']):
    for s, sub in enumerate(subjects):
        subject_runs = [r==sub for r in metadata['subjects']]
        roi_mask = rois[s, roi, :]
        patterns = betas[subject_runs][:, :, roi_mask]
        ds = Dataset(
            measurements=patterns.reshape(-1, roi_mask.sum()),
            descriptors=dict(sub=sub, roi=region_name),
            obs_descriptors=dict(
                run=numpy.repeat(numpy.arange(N_RUNS), len(conditions)),
                condition=numpy.tile(conditions, N_RUNS)
            )
        )




## prec from noice
## dataset per subject and roi
## RDMs per roi



