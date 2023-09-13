""""

The 214 repeated images are spread out over 100 runs in 10 sessions (2 images per run).

## TODO

- [ ] both hemis
- [ ] coco, scenes, imagenet or all
- [x] otherwise, get betas per run
- [ ] whitening
- [ ] model each individual rep within run

"""
## suppress warnings on nibabel
# pyright: reportPrivateImportUsage=false 
from os.path import expanduser, join, isfile
import itertools
import datalad.api as datalad
import nibabel, pandas, numpy, tqdm
from sklearn import neighbors as skl_neighbors
from scipy.interpolate import pchip

#description: https://bold5000-dataset.github.io/website/overview.html
# 112 images repeated four times, and one image repeated three times

## this will setup a local copy of the dataset, but only downloads text files
openneuro_id = 1499
your_data_dir = expanduser('~/data')
dataset_dir = join(your_data_dir, f'ds00{openneuro_id}')
fmriprep_dir = join(dataset_dir, 'derivatives', 'fmriprep')


# dl = datalad.clone(
#     source=f'///openneuro/ds00{openneuro_id}', 
#     path=dataset_dir,
#     description='BOLD5000 v1' 
# )
## download fmriprep output for subject 1; ~49GB, 12h to download
# dl.get('derivatives/fmriprep/sub-CSI1/')

# meta_fpath = join(root_dir, 'sub-04/ses-1/func/sub-04_ses-1_task-motor_run-01_bold.json')
# with open(meta_fpath) as fhandle:
#     metadata = json.load(fhandle)
# TR = metadata['RepetitionTime']
tr = 2.0
block_dur = 1.0

STANDARD_HRF = numpy.load('demos/hrf.npy')
STANDARD_TR = 0.1
hrf = numpy.convolve(STANDARD_HRF, numpy.ones(int(block_dur/STANDARD_TR)))

## timepoints in block
timepts_block = numpy.arange(0, int((hrf.size-1)*STANDARD_TR), tr)

# resample to desired TR
hrf = pchip(numpy.arange(hrf.size)*STANDARD_TR, hrf)(timepts_block)
hrf = hrf / hrf.max()

print('starting neighbor search')

## get coords of vertices
fpath_anat = join(fmriprep_dir, 'sub-CSI1', 'anat', f'sub-CSI1_T1w_inflated.L.surf.gii')
img_anat = nibabel.load(fpath_anat)
coords = img_anat.agg_data('pointset')
nn = skl_neighbors.NearestNeighbors(radius=5)
adjacency = nn.fit(coords).radius_neighbors_graph(coords).tolil() # sparse

#raise ValueError

"""388s 194 volumes

"""

n_obs = (112 * 4) + (1 * 3) 

sessions_runs = list(itertools.product(range(1, 15+1), range(1, 10+1)))
#sessions_runs = [(1, 1)]
all_events = []
all_betas = []
for (s, r) in tqdm.tqdm(sessions_runs): ## 2m35s
    ses_raw_dir = join(dataset_dir, 'sub-CSI1', f'ses-{s:02}', 'func')
    evt_fpath = join(ses_raw_dir, 
        f'sub-CSI1_ses-{s:02}_task-5000scenes_run-{r:02}_events.tsv')
    
    if not isfile(evt_fpath):
        print(f'Missing run: session {s} run {r}')
        continue

    events_df = pandas.read_csv(evt_fpath, sep='\t')
    events_df['trial_type'] = events_df['ImgName']


    ses_fmriprep_dir = join(fmriprep_dir, 'sub-CSI1', f'ses-{s:02}', 'func')
    bold_fpath = join(ses_fmriprep_dir,
        f'sub-CSI1_ses-{s:02}_task-5000scenes_run-{r:02}_bold_space-fsnative.L.func.gii')
    data = nibabel.load(bold_fpath).agg_data() # (135186, 194) # 100MB


    ## make design matrix
    conditions = events_df.trial_type.unique()
    n_vols = data.shape[-1]
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

    # pct signal change
    data = data / data.mean(axis=0)

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

    # gather betas if in rep conditions
    conds_df = events_df.drop_duplicates(subset=['ImgName'])
    relevant_conds = conds_df.ImgType.str.contains('rep_').values
    all_events += [conds_df[relevant_conds]]
    all_betas += [betas[relevant_conds, :]]
all_df = pandas.concat(all_events).reset_index()
## keep only columns we may need
all_df = all_df[['Sess', 'Run', 'trial_type', 'ImgType', 'Response']]
all_betas_a = numpy.vstack(all_betas)

## save point
all_df.to_csv('events.csv')
numpy.save('betas.npy', all_betas_a)


"""from exploration

In [44]: df[df.ImgType=='rep_coco'].ImgName.value_counts().size
Out[44]: 45

In [45]: df[df.ImgType=='rep_scenes'].ImgName.value_counts().size
Out[45]: 23

In [46]: df[df.ImgType=='rep_imagenet'].ImgName.value_counts().size
Out[46]: 45

"""


"""from paper methods:

In order to examine the effect of image repetition, we randomly 
selected 112 of the 4,916 distinct images to be shown four times 
and one image to be shown three times to each participant. 
These 113 images were selected such that the image dataset breakdown 
was proportionally to that of the 4,916 distinct images. 
Specifically, 1/5 of the images were Scene images, 2/5 of the images were COCO images, 
2/5 of the images were ImageNet images. When these image repetitions are considered, 
we have a total of 5,254 image presentations shown to each participant 
(4,803 distinct images +4 × 112 repeated images +3 × 1 repeated image). 
For CSI3 and CSI4, a small number of repetitions varied from 2–5 times.

----

The following image presentation details apply for each run, each session, 
and each participant. A slow event-related design was implemented for stimulus 
presentation in order to isolate the blood oxygen level dependent (BOLD) signal 
for each individual image trial. At the beginning and end of each run, 
centered on a blank, black screen, a fixation cross was shown for 6 sec and 12 sec, 
respectively. Following the initial fixation cross, all 37 stimuli were shown sequentially. 
Each image was presented for 1 sec followed by a 9 sec fixation cross. Given that each run 
contains 37 stimuli, there was a total of 370 sec of stimulus presentation plus fixation. 
Including the pre- and post-stimulus fixations, there were a total of 388 sec 
(6 min 28 sec) of data acquired in each run.

For each stimulus image shown, each participant performed a valence judgment task, 
responding with how much they liked the image using the metric: “like”, “neutral”, 
“dislike”. Responses were collected during the 9 sec interval comprising the interstimulus 
fixation, that is, subsequent to the stimulus presentation, and were made by 
pressing buttons attached to an MRI-compatible response glove on their dominant hand 
(right for all participants).
"""