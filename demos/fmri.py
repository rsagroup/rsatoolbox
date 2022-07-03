from os.path import expanduser
import datalad.api as datalad
from numpy import atleast_1d

## this will setup a local copy of the dataset, but only downloads text files
dl = datalad.clone(
    '///openneuro/ds004044', 
    expanduser('~/data/ds004044'),
    description='Ma et al 2022 Somatotopy'
)

## download fmriprep output for subject 4, 1.13GB, 15mins
dl.get('derivatives/fmriprep/sub-04/')

root 'sub-04/ses-1/func/sub-04_ses-1_task-motor_run-04_events.tsv'
derivatives/fmriprep 'sub-04_ses-1_task-motor_run-1_space-T1w_desc-preproc_bold_denoised.nii.gz'


## gather fmriprep datasets
## fmri dataset: 
""""
- space desc
- atlas option
- events -> volume-wise event descriptor + cond
- option to add predictors + hrf

- transform to voxels x conditions
- one option to keep runs separate but collapse repetitions

"""