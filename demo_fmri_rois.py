
from os.path import expanduser
from rsatoolbox.io.fmriprep import find_fmriprep_runs
import numpy, pandas, nibabel
from os.path import join
import matplotlib.pyplot as plt
from nilearn.plotting import plot_stat_map


THRESHOLD = 20 ## hopw many runs need to overlap for a voxel to be selected

data_dir = expanduser('~/data/rsatoolbox/mur32')

runs = find_fmriprep_runs(data_dir, tasks=['main'])
runs = [run for run in runs if run.boldFile.sub != '02']

## basic metadata
run = runs[0]

an_img = nibabel.load(run.boldFile.fpath)
x, y, z, n_vols = an_img.shape
n_voxels = x*y*z
affine = an_img.affine
tr = run.get_meta()['RepetitionTime']

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
counts = numpy.zeros((len(region_names), n_voxels))
for r, region_name in enumerate(region_names):
    print(f'mapping {region_name}..')

    for run in runs:

        aparc = run.boldFile.get_mri_sibling(desc='aparcaseg', suffix='dseg')
        aparc_data = aparc.get_data()

        for hemi in ('r', 'l'):
            full_name = f'ctx-{hemi}h-{region_name}'
            matches = lut_df[lut_df['name']==full_name]
            assert len(matches) == 1, f'None or multiple matches for {full_name}'
            region_id = matches['index'].values[0]
            mask = (aparc_data == float(region_id)).ravel()
            counts[r, :] += (mask*1)

    vol = counts[r, :].reshape(x,y,z)
    img = nibabel.nifti1.Nifti1Image(vol, affine)
    fig = plt.figure()
    plot_stat_map(img, cmap='viridis', title=region_name)
    plt.savefig(f'{region_name}_overlap.png')
    plt.close(fig)

with open('rois_names.txt', 'w') as fhandle:
    fhandle.writelines([n+'\n' for n in region_names])      
numpy.save('rois.npy', counts)

