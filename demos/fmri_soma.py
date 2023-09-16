""""

## this demo todos:
- loop over runs
- ciftify/sub/native -> aparcaseg
- polynomials
- pct signal change


"Motor tasks of several major body parts, including toe, ankle, left leg, right leg, finger, wrist, forearm, upper arm, jaw, lip, tongue, eye"

"""
from os.path import expanduser, join
import datalad.api as datalad
import json
import numpy, nibabel, pandas
from scipy.interpolate import pchip
from rsatoolbox.data.dataset import Dataset
from rsatoolbox.rdm.calc import calc_rdm
from rsatoolbox.vis.rdm_plot import show_rdm

## this will setup a local copy of the dataset, but only downloads text files
root_dir = expanduser('~/data/ds004044')
dl = datalad.clone(
    source='///openneuro/ds004044', 
    path=root_dir,
    description='Ma et al 2022 Somatotopy'
)

## download fmriprep output for subject 4, 1.13GB, 15mins
dl.get('derivatives/fmriprep/sub-04/')
dl.get('derivatives/melodic/sub-04/')
dl.get('derivatives/ciftify/sub-04/')

meta_fpath = join(root_dir, 'sub-04/ses-1/func/sub-04_ses-1_task-motor_run-01_bold.json')
with open(meta_fpath) as fhandle:
    metadata = json.load(fhandle)
tr = metadata['RepetitionTime']

events_fpath = join(root_dir, 'sub-04/ses-1/func/sub-04_ses-1_task-motor_run-04_events.tsv')
events_df = pandas.read_csv(events_fpath, sep='\t')

bold_fpath = join(
    root_dir, 'derivatives/fmriprep/sub-04/',
    'sub-04_ses-1_task-motor_run-1_space-T1w_desc-preproc_bold_denoised.nii.gz'
)
bold_data = nibabel.load(bold_fpath).get_fdata()


## i'm guessing this. baseline + conditions mentioned in readme
labels = ['baseline', 'toe', 'ankle', 'left leg', 'right leg', 'finger', 'wrist', 
    'forearm', 'upper arm', 'jaw', 'lip', 'tongue', 'eye']
ds = Dataset(
    measurements=betas,
    obs_descriptors=dict(label=labels)
)
rdms = calc_rdm(ds, method='correlation')
rdms.pattern_descriptors['label'] = labels
show_rdm(rdms, pattern_descriptor='labels')
