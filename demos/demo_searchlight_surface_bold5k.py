""""

The 214 repeated images are spread out over 100 runs in 10 sessions (2 images per run).

GLMsingle valid?

"""
## suppress warnings on nibabel
# pyright: reportPrivateImportUsage=false 
from os.path import expanduser, join
import datalad.api as datalad
import json, glob
import nibabel, pandas

#description: https://bold5000-dataset.github.io/website/overview.html
# 112 images repeated four times, and one image repeated three times

## this will setup a local copy of the dataset, but only downloads text files
openneuro_id = 1499
your_data_dir = expanduser('~/data')
dataset_dir = join(your_data_dir, f'ds00{openneuro_id}')
fmriprep_dir = join(dataset_dir, 'derivatives', 'fmriprep')
dl = datalad.clone(
    source=f'///openneuro/ds00{openneuro_id}', 
    path=dataset_dir,
    description='BOLD5000 v1'
)


## download fmriprep output for subject 1; ~49GB, 12h to download
dl.get('derivatives/fmriprep/sub-CSI1/')

# meta_fpath = join(root_dir, 'sub-04/ses-1/func/sub-04_ses-1_task-motor_run-01_bold.json')
# with open(meta_fpath) as fhandle:
#     metadata = json.load(fhandle)
# TR = metadata['RepetitionTime']

fpath_anat = join(fmriprep_dir, 'sub-CSI1', 'anat', f'sub-CSI1_T1w_inflated.L.surf.gii')
img_anat = nibabel.load(fpath_anat)
coords = img_anat.agg_data('pointset')


"""388s 194 volumes



"""
all_runs = []
for s in range(1, 15+1):
    ses_dir = join(dataset_dir, 'sub-CSI1', f'ses-{s:02}', 'func')
    ses_fmriprep_dir = join(fmriprep_dir, 'sub-CSI1', f'ses-{s:02}', 'func')
    run_evt_fpaths = glob.glob(join(ses_dir, '*_task-5000scenes_*_events.tsv'))
    for fpath in run_evt_fpaths:
        r = int(fpath.split('_')[-2][-2:])
        run_df = pandas.read_csv(fpath, sep='\t')
        bold_fname = f'sub-CSI1_ses-{s:02}_task-5000scenes_run-{r:02}_bold_space-fsnative.L.func.gii'
        bold_fpath = join(ses_fmriprep_dir, bold_fname)
        data = nibabel.load(bold_fpath).agg_data() # (135186, 194) # 100MB
        all_runs.append(run_df)
        raise ValueError
df = pandas.concat(all_runs)

## zip ses/run assume 10 runs, exit if run not there

""" reps by cat
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