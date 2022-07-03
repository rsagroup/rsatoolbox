from os.path import expanduser, join
import datalad.api as datalad
import json
import nibabel, pandas

## this will setup a local copy of the dataset, but only downloads text files
root_dir = expanduser('~/data/ds004044')
dl = datalad.clone(
    source='///openneuro/ds004044', 
    path=root_dir,
    description='Ma et al 2022 Somatotopy'
)

## download fmriprep output for subject 4, 1.13GB, 15mins
dl.get('derivatives/fmriprep/sub-04/')

meta_fpath = join(root_dir, 'sub-04/ses-1/func/sub-04_ses-1_task-motor_run-01_bold.json')
with open(meta_fpath) as fhandle:
    metadata = json.load(fhandle)
TR = metadata['RepetitionTime']

events_fpath = join(root_dir, 'sub-04/ses-1/func/sub-04_ses-1_task-motor_run-04_events.tsv')
df = pandas.read_csv(events_fpath, sep='\t')

bold_fpath = join(
    root_dir, 'derivatives/fmriprep/sub-04/',
    'sub-04_ses-1_task-motor_run-1_space-T1w_desc-preproc_bold_denoised.nii.gz'
)
bold_data = nibabel.load(bold_fpath).get_fdata()


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