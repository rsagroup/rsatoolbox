"""Analysis script for mur32 to prepare data for fmri demo

Two ways to store this data
1) complete statmaps; approx 2.3GB (I'll do this for now as it seems more intuitive for students)
2) only roi voxels; (concatenate rois) 1/10 the size but need separate file/subject

"""
from os.path import expanduser, join
import json, warnings
from rsatoolbox.io.fmriprep import find_fmriprep_runs
import numpy, pandas
import nibabel
from nibabel.nifti1 import Nifti1Image
from nilearn.glm.first_level import make_first_level_design_matrix
from nilearn.glm.first_level import FirstLevelModel


def calc_urun(ses: str, run: str) -> int:
    """Generate unique run index, disregarding sessions

    Args:
        run (str): run name, e.g. "02"
        ses (str): ses name, e.g. "02"

    Returns:
        int: unique number for run
    """
    return int(run) + ((int(ses) - 1) * 3)

data_dir = expanduser('~/data/rsatoolbox/mur32')
out_dir = 'fmri_data'

print('indexing fmriprep bold runs..')
runs = find_fmriprep_runs(data_dir, tasks=['main'])

## remove subject 2
runs = [run for run in runs if run.boldFile.sub != '02']
subjects = sorted(set([run.sub for run in runs]))

## take a single run to get some basic metadata
run = runs[0]
an_img = nibabel.load(run.boldFile.fpath)
x, y, z, n_vols = an_img.shape
n_voxels = x*y*z
affine = an_img.affine

## prepare basics for design matrix
tr = run.get_meta()['RepetitionTime'] ## TR in seconds
frame_times = numpy.linspace(0, tr*(n_vols-1), n_vols) ## [0, 2, 4] onsets of scans in seconds
trial_types = run.get_events().trial_type.unique()
conditions = sorted(filter(lambda t: '_' in t, trial_types))


## access the fmriprep look-up-table for aparc 
lut_fpath = join(data_dir, 'derivatives', 'fmriprep',
    'desc-aparcaseg_dseg.tsv')
lut_df = pandas.read_csv(lut_fpath, sep='\t')

region_names = [
    'cuneus',
    'pericalcarine',
    'lingual',
    'fusiform',
    'lateraloccipital',
    'inferiortemporal',
    'inferiorparietal'
]
rois = numpy.zeros((len(subjects), len(region_names), n_voxels), dtype=bool)
for s, sub in enumerate(subjects):
    for roi, region_name in enumerate(region_names):
        print(f'for subject {sub} mapping {region_name} ..')

        subject_run = [run for run in runs if run.sub == sub][0]
        aparc = subject_run.boldFile.get_mri_sibling(desc='aparcaseg', suffix='dseg')
        aparc_data = aparc.get_data()

        for hemi in ('r', 'l'):
            full_name = f'ctx-{hemi}h-{region_name}'
            matches = lut_df[lut_df['name']==full_name]
            assert len(matches) == 1, f'None or multiple matches for {full_name}'
            region_id = matches['index'].values[0]
            roi_mask = (aparc_data == float(region_id)).ravel()
            rois[s, roi, :][roi_mask] = True

## Loop subjects
print('Reserving memory..')
betas = numpy.full([len(runs), len(conditions), n_voxels], numpy.nan)
resids = numpy.full([len(runs), n_vols, n_voxels], numpy.nan)
for r, run in enumerate(runs):
    urun = calc_urun(run.boldFile.ses, run.run)
    print(f'Fitting GLM for sub {run.sub} urun {urun} ses {run.boldFile.ses} run {run.run}..')

    with warnings.catch_warnings(action='ignore'):
        design_matrix = make_first_level_design_matrix(
            frame_times,
            run.get_events(),
            drift_model='polynomial',
            drift_order=3
        )

    sub_mask = numpy.any(rois[s, :, :], axis=0).reshape(x,y,z)
    glm = FirstLevelModel(
        t_r=tr,
        ## no speedup from masking but smaller filesize probs
        mask_img=Nifti1Image(sub_mask.astype(float), affine=affine),
        minimize_memory=False,  ## to enable residuals
        signal_scaling=0,       ## 0 = psc, False = off
        n_jobs=-3               ## all but two
    )
    glm.fit([run.boldFile.fpath], design_matrices=design_matrix)
    
    for c, condition in enumerate(conditions):
        beta_img = glm.compute_contrast(condition, output_type='effect_size')
        betas[r, c, :] = beta_img.get_fdata().ravel()

    resid_img = glm.residuals[0]
    resids[r, :, :] = resid_img.get_fdata().reshape(n_voxels, n_vols).T

print('Compressing..')
numpy.savez_compressed(
    join(out_dir, 'data.npz'),
    rois=rois,
    betas=betas,
    resids=resids,
)
with open(join(out_dir, 'meta.json'), 'w') as fhandle:
    json.dump(
        dict(
            region_names=region_names,
            conditions=conditions,
            subjects=[run.sub for run in runs],
            uruns = [calc_urun(run.boldFile.ses, run.run) for run in runs],
            runs=[run.run for run in runs],
            sessions=[run.boldFile.ses for run in runs],
        ),
        fhandle
    )
