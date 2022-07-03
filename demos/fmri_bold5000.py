from os.path import expanduser, join
import datalad.api as datalad
import json
import nibabel, pandas

#description: https://bold5000-dataset.github.io/website/overview.html
# 112 images repeated four times, and one image repeated three times

## this will setup a local copy of the dataset, but only downloads text files
openneuro_id = 1499
your_data_dir = expanduser('~/data')
dataset_dir = join(your_data_dir, f'ds00{openneuro_id}')
dl = datalad.clone(
    source=f'///openneuro/ds00{openneuro_id}', 
    path=dataset_dir,
    description='BOLD5000 v1'
)


## download fmriprep output for subject 1, session 1; 3.78GB
dl.get('derivatives/fmriprep/sub-CSI1/ses-01/')

# meta_fpath = join(root_dir, 'sub-04/ses-1/func/sub-04_ses-1_task-motor_run-01_bold.json')
# with open(meta_fpath) as fhandle:
#     metadata = json.load(fhandle)
# TR = metadata['RepetitionTime']

# events_fpath = join(dataset_dir, 'sub-04/ses-1/func/sub-04_ses-1_task-motor_run-04_events.tsv')
# df = pandas.read_csv(events_fpath, sep='\t')

# bold_fpath = join(
#     dataset_dir, 'derivatives/fmriprep/sub-04/',
#     'sub-04_ses-1_task-motor_run-1_space-T1w_desc-preproc_bold_denoised.nii.gz'
# )
# bold_data = nibabel.load(bold_fpath).get_fdata()


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