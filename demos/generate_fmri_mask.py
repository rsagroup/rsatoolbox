"""
Script to generate standard space mask
"""
from os.path import expanduser
from rsatoolbox.io.fmriprep import find_fmriprep_runs
import numpy, nibabel

MIN_OVERLAP = 60 ## at least 10 subjects (x6 runs) overlap

# Now we define the path to the BIDS-root data directory
data_dir = expanduser('~/data/rsatoolbox/mur32')
runs = find_fmriprep_runs(data_dir, tasks=['main'])
runs = [run for run in runs if run.boldFile.sub != '02']

# template vals
run0 = runs[0]
brainMask = run0.boldFile.get_mri_sibling(desc='brain', suffix='mask')
template_img = nibabel.load(brainMask.fpath)
overlap = numpy.zeros(run0.get_mask().shape, dtype=int)
df = run0.get_parcellation_labels()
label = 'ctx-rh-inferiortemporal'
roi_index = df[df.name==label].index.values[0] # 2009

for run in runs:
    print('.')
    parc_ix_3d = run.get_parcellation()
    overlap = overlap + (parc_ix_3d == roi_index).astype(int)

mask = overlap > 60
print(f'Number of voxels selected: {mask.sum()}')
mask_3d = mask.reshape(run0.get_mask().shape)
fpath = f'demos/{label}.nii.gz'
mask_img = nibabel.Nifti1Image(mask_3d, template_img.affine, template_img.header)
nibabel.save(mask_img, fpath)
