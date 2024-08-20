"""Run code that generates the various data objects to be used

See also demos/prepare_demo_fmri_patterns.py
"""
from os.path import expanduser, join
import json
import warnings
import numpy
import pandas
import nibabel
from nibabel.nifti1 import Nifti1Image
from nilearn.glm.first_level import make_first_level_design_matrix
from nilearn.glm.first_level import FirstLevelModel
from rsatoolbox.io.fmriprep import find_fmriprep_runs
from rsatoolbox.data.dataset import Dataset
from rsatoolbox.data.noise import prec_from_residuals
from rsatoolbox.rdm.calc import calc_rdm

## subset selection
sub = '07'
region_name = 'fusiform'
data_dir = expanduser('~/data/rsatoolbox/mur32')
out_dir = join(data_dir, 'derivatives', 'figure1')


def calc_urun(ses: str, run: str) -> int:
    """Generate unique run index, disregarding sessions

    Args:
        run (str): run name, e.g. "02"
        ses (str): ses name, e.g. "02"

    Returns:
        int: unique number for run
    """
    return int(run) + ((int(ses) - 1) * 3) - 1


print('indexing fmriprep bold runs..')
runs = find_fmriprep_runs(data_dir, tasks=['main'])

# remove subject 2
runs = [run for run in runs if run.boldFile.sub == '12']
subjects = sorted(set([run.sub for run in runs]))
N_RUNS = 6

# take a single run to get some basic metadata
run = runs[0]
an_img = nibabel.load(run.boldFile.fpath)
x, y, z, n_vols = an_img.shape
n_voxels = x*y*z
affine = an_img.affine

# prepare basics for design matrix
degrees_of_freedom = 0  # to be assigned later
tr = run.get_meta()['RepetitionTime']  # TR in seconds
# [0, 2, 4] onsets of scans in seconds
frame_times = numpy.linspace(0, tr*(n_vols-1), n_vols)
trial_types = run.get_events().trial_type.unique()
conditions = sorted(filter(lambda t: '_' in t, trial_types))


# access the fmriprep look-up-table for aparc
lut_fpath = join(data_dir, 'derivatives', 'fmriprep',
                 'desc-aparcaseg_dseg.tsv')
lut_df = pandas.read_csv(lut_fpath, sep='\t')

region_names = [
    'fusiform',
    'lateraloccipital',
]
rois = numpy.zeros((len(subjects), len(region_names), n_voxels), dtype=bool)
data = dict()
for s, sub in enumerate(subjects):
    for roi, region_name in enumerate(region_names):
        print(f'for subject {sub} mapping {region_name} ..')

        subject_run = [run for run in runs if run.sub == sub][0]
        aparc = subject_run.boldFile.get_mri_sibling(
            desc='aparcaseg', suffix='dseg')
        aparc_data = aparc.get_data()

        for hemi in ('r', 'l'):
            full_name = f'ctx-{hemi}h-{region_name}'
            matches = lut_df[lut_df['name'] == full_name]
            msg = f'None or multiple matches for {full_name}'
            assert len(matches) == 1, msg
            region_id = matches['index'].values[0]
            roi_mask = (aparc_data == float(region_id)).ravel()
            rois[s, roi, :][roi_mask] = True

        roi_size = rois[s, roi, :].sum()

        data[f'betas_sub-{sub}_{region_name}'] = numpy.full(
            [N_RUNS, len(conditions), roi_size], numpy.nan)
        data[f'resids_sub-{sub}_{region_name}'] = numpy.full(
            [N_RUNS, n_vols, roi_size], numpy.nan)

for run in runs:
    r = calc_urun(run.boldFile.ses, run.run)
    print(f'Fitting GLM for sub {run.sub} urun {r} '
          f'ses {run.boldFile.ses} run {run.run}..')

    with warnings.catch_warnings(action='ignore'):
        design_matrix = make_first_level_design_matrix(
            frame_times,
            run.get_events(),
            drift_model='polynomial',
            drift_order=3
        )
    if degrees_of_freedom == 0:
        degrees_of_freedom = design_matrix.shape[1]
    else:
        # make sure dof is the same throughout
        assert design_matrix.shape[1] == degrees_of_freedom

    subject_rois = rois[subjects.index(run.sub), :, :]
    sub_mask = numpy.any(subject_rois, axis=0).reshape(x, y, z)
    glm = FirstLevelModel(
        t_r=tr,
        # no speedup from masking but smaller filesize probs
        mask_img=Nifti1Image(sub_mask.astype(float), affine=affine),
        minimize_memory=False,  # to enable residuals
        signal_scaling=0,       # 0 = psc, False = off
        n_jobs=-3               # all but two
    )
    glm.fit([run.boldFile.fpath], design_matrices=design_matrix)

    resid_img = glm.residuals[0]
    run_resids = resid_img.get_fdata().reshape(n_voxels, n_vols)

    for roi, region_name in enumerate(region_names):
        tag = f'resids_sub-{run.sub}_{region_name}'
        roi_mask = rois[subjects.index(run.sub), roi, :]
        data[tag][r, :, :] = run_resids[roi_mask, :].T

    for c, condition in enumerate(conditions):
        beta_img = glm.compute_contrast(condition, output_type='effect_size')
        betas = beta_img.get_fdata().ravel()

        for roi, region_name in enumerate(region_names):
            tag = f'betas_sub-{run.sub}_{region_name}'
            roi_mask = rois[subjects.index(run.sub), roi, :]
            data[tag][r, c, :] = betas[roi_mask]

print('Compressing..')
numpy.savez_compressed(
    join(out_dir, 'data.npz'),
    rois=rois,
    **data,
)
with open(join(out_dir, 'meta.json'), 'w') as fhandle:
    json.dump(
        dict(
            region_names=region_names,
            conditions=conditions,
            subjects=subjects,
            degrees_of_freedom=degrees_of_freedom
        ),
        fhandle
    )

### demo section

betas = data[f'betas_sub-{sub}_{region_name}']
patterns = betas.reshape(-1, betas.shape[-1])

# create an rsatoolbox Dataset with the patterns
ds = Dataset(
    measurements=patterns,
    descriptors=dict(sub=sub, roi=region_name),
    obs_descriptors=dict(
        run=numpy.repeat(numpy.arange(N_RUNS), len(conditions)),
        condition=numpy.tile(conditions, N_RUNS)
    )
)
ds.save(join(out_dir, 'dataset.h5'), overwrite=True)

runwise_prec_matrix = []
resids = data[f'resids_sub-{sub}_{region_name}']
for r in range(N_RUNS):
    runwise_prec_matrix.append(
        prec_from_residuals(
            resids[r, :, :],
            dof=degrees_of_freedom,
            method='shrinkage_diag'
        )
    )
numpy.save(join(out_dir, 'noise.npy'), runwise_prec_matrix)

# calculate crossnobis RDMs from the patterns and precision matrices
rdms = calc_rdm(
    dataset=ds,
    noise=runwise_prec_matrix,
    method='crossnobis',
    descriptor='condition',
    cv_descriptor='run',
)

rdms.descriptors['noise'] # saves memory
rdms.save(join(out_dir, 'rdms.h5'), overwrite=True)