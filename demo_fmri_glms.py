"""Analysis script for mur32 to prepare data for fmri demo


Two ways to store this data
1) complete statmaps; approx 2.3GB (I'll do this for now as it seems more intuitive for students)
2) only roi voxels; (concatenate rois) 1/10 the size but need separate file/subject

"""
from os.path import expanduser, join
from rsatoolbox.io.fmriprep import find_fmriprep_runs
import numpy, pandas, matplotlib.pyplot
import nilearn, nibabel
import matplotlib.pyplot as plt
from nilearn.glm.first_level import make_first_level_design_matrix
from nilearn.plotting import plot_design_matrix, plot_stat_map, plot_glass_brain
from nilearn.glm.first_level import FirstLevelModel

data_dir = expanduser('~/data/rsatoolbox/mur32')
out_dir = 'fmri_data'

print('indexing fmriprep bold runs..')
runs = find_fmriprep_runs(data_dir, tasks=['main'])

runs = [run for run in runs if run.boldFile.sub != '02']
subjects = sorted(set([run.sub for run in runs]))

run = runs[0]
tr = run.get_meta()['RepetitionTime'] ## TR in seconds
an_img = nibabel.load(run.boldFile.fpath)
x, y, z, n_vols = an_img.shape
n_voxels = x*y*z
affine = an_img.affine

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
    for r, region_name in enumerate(region_names):
        print(f'for subject {sub} mapping {region_name} ..')

        subject_run = [run for run in runs if run.sub == sub][0]
        aparc = subject_run.boldFile.get_mri_sibling(desc='aparcaseg', suffix='dseg')
        aparc_data = aparc.get_data()

        for hemi in ('r', 'l'):
            full_name = f'ctx-{hemi}h-{region_name}'
            matches = lut_df[lut_df['name']==full_name]
            assert len(matches) == 1, f'None or multiple matches for {full_name}'
            region_id = matches['index'].values[0]
            mask = (aparc_data == float(region_id)).ravel()
            rois[s, r, :][mask] = True


raise ValueError

frame_times = numpy.linspace(0, tr*(n_vols-1), n_vols) ## [0, 2, 4] onsets of scans in seconds
design_matrix = make_first_level_design_matrix(frame_times, run.get_events(),
    drift_model='polynomial', drift_order=3)

## Loop subjects
# for s, sub in enumerate(subjects):
## make all-roi mask
## run glm
## apply contrasts?
## select roi data





## unpack a roi
vol = counts[r, :].reshape(x,y,z)
img = nibabel.nifti1.Nifti1Image(vol, affine)

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
    

    ### loop over regions here

    patterns = []
    for cond in conditions:
        z_map = glm.compute_contrast(f'{cond} - baseline', output_type='z_score')


        numpy.asarray(z_map.get_fdata()[it_mask])

    

    glm.residuals[0].get_fdata()[it_mask].T,

