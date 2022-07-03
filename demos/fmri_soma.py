""""

## demo todo:
- loop over runs

## functionality todo's:
- function which gather fmriprep datasets
- fmri dataset (pre-patterns)
- SpatialDataSet (for patterns)

- space desc
- atlas option
- events -> volume-wise event descriptor + cond
- option to add predictors + hrf

- transform to voxels x conditions
- one option to keep runs separate but collapse repetitions


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

## make mask
mean_bold = bold_data.mean(axis=3)
mask = mean_bold>mean_bold.mean()*1.5
data = bold_data[mask, :]
#plt.imshow(mean_bold[:,:, 30])


## make design matrix
n_vols = bold_data.shape[-1]
block_dur = numpy.median(events_df.duration)

## convolve a standard HRF to the block shape in the design
STANDARD_HRF = numpy.load('demos/hrf.npy')
STANDARD_TR = 0.1
hrf = numpy.convolve(STANDARD_HRF, numpy.ones(int(block_dur/STANDARD_TR)))

## timepoints in block (32x)
timepts_block = numpy.arange(0, int((hrf.size-1)*STANDARD_TR), tr)

# resample to desired TR
hrf = pchip(numpy.arange(hrf.size)*STANDARD_TR, hrf)(timepts_block)
hrf = hrf / hrf.max()

## make design matrix
conditions = events_df.trial_type.unique()
dm = numpy.zeros((n_vols, conditions.size))
all_times = numpy.linspace(0, tr*(n_vols-1), n_vols)
hrf_times = numpy.linspace(0, tr*(len(hrf)-1), len(hrf))
for c, condition in enumerate(conditions):
    onsets = events_df[events_df.trial_type == condition].onset.values
    yvals = numpy.zeros((n_vols))
    # loop over blocks
    for o in onsets:
        # interpolate to find values at the data sampling time points
        f = pchip(o + hrf_times, hrf, extrapolate=False)(all_times)
        yvals = yvals + numpy.nan_to_num(f)
    dm[:, c] = yvals

## wdata, wdesign = whiten_data(data, design) ## adds polynomials
## pdata = wdata / wdata.mean(axis=0)

## least square fitting
# The matrix addition is equivalent to concatenating the list of data and the list of
# design and fit it all at once. However, this is more memory efficient.
design = [dm]
data = [data.T]
X = numpy.vstack(design)
X = numpy.linalg.inv(X.T @ X) @ X.T

betas = 0
start_col = 0
for run in range(len(data)):
    n_vols = data[run].shape[0]
    these_cols = numpy.arange(n_vols) + start_col
    betas += X[:, these_cols] @ data[run]
    start_col += data[run].shape[0]

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
