
from os.path import expanduser
from rsatoolbox.io.fmriprep import find_fmriprep_runs, make_design_matrix
from rsatoolbox.data.dataset import Dataset, merge_datasets
from rsatoolbox.rdm.rdms import concat
from rsatoolbox.vis import show_rdm
from rsatoolbox.rdm.calc import calc_rdm
from rsatoolbox.data.noise import prec_from_residuals
import numpy, pandas, matplotlib.pyplot
import nilearn, nibabel
import matplotlib.pyplot as plt

from nilearn.glm.first_level import make_first_level_design_matrix
from nilearn.plotting import plot_design_matrix, plot_stat_map, plot_glass_brain
from nilearn.glm.first_level import FirstLevelModel

data_dir = expanduser('~/data/rsatoolbox/mur32')

runs = find_fmriprep_runs(data_dir, tasks=['main'])

runs = [run for run in runs if run.boldFile.sub != '02']

run = runs[0]
dims = run.get_data(masked=True).shape ## bold timeseries: x * y * z * volumes
n_vols = dims[-1]

tr = run.get_meta()['RepetitionTime'] ## TR in seconds




frame_times = numpy.linspace(0, tr*(n_vols-1), n_vols) ## [0, 2, 4] onsets of scans in seconds
design_matrix = make_first_level_design_matrix(frame_times, run.get_events(),
    drift_model='polynomial', drift_order=3)
plot_design_matrix(design_matrix)


mask_fpath = run.boldFile.get_mri_sibling(desc='brain', suffix='mask').fpath
glm = FirstLevelModel(
    t_r=2.0,
    mask_img=False,
    minimize_memory=False,
    signal_scaling=False,
)
glm.fit([run.boldFile.fpath], design_matrices=design_matrix)


conditions = [c for c in design_matrix.columns if ('image_' in c) or ('text_' in c)]


contrast_val = ['image_glove - baseline']
z_map = glm.compute_contrast(contrast_val, output_type='z_score')



from nilearn.plotting import plot_stat_map
plot_stat_map(z_map, threshold=3) # bg_img=mean_img


mask_img = nibabel.load('ctx-rh-inferiortemporal.nii.gz')
it_mask = mask_img.get_fdata().astype(bool)
it_mask.sum()
it_mask.shape
z_map.get_fdata().shape


it_pattern = z_map.get_fdata()[it_mask]
numpy.count_nonzero(it_pattern)
it_pattern.size




run_datasets = []
run_noise_sets = []

for run in runs:
    design_matrix = make_first_level_design_matrix(
        frame_times,
        run.get_events(),
        drift_model='polynomial',
        drift_order=3
    )
    glm = FirstLevelModel( ## necessary to redefine this? docs do not say if fit is idempotent
        t_r=2.0,
        mask_img=False,
        minimize_memory=False,
        signal_scaling=False,
    )
    glm.fit([run.boldFile.fpath], design_matrices=design_matrix)
    
    patterns = []
    for cond in conditions:
        z_map = glm.compute_contrast(f'{cond} - baseline', output_type='z_score')
        patterns.append(z_map.get_fdata()[it_mask])

    run_datasets.append(
        Dataset(
            measurements=numpy.asarray(patterns),
            descriptors=run.get_dataset_descriptors(),
            obs_descriptors=dict(trial_type=conditions),
        )
    )

    run_noise_sets.append(
        Dataset(
            measurements=glm.residuals[0].get_fdata()[it_mask].T,
            descriptors=run.get_dataset_descriptors(),
        )
    )
