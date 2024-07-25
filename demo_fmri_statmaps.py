"""Script version of statmap-based fmri demo

This approach: store full statmaps
Other approach: store only masked roi patterns

"""
from os.path import expanduser, join
import json, warnings
import numpy
from nibabel.nifti1 import Nifti1Image
from rsatoolbox.data.dataset import Dataset
from rsatoolbox.data.noise import prec_from_residuals
from rsatoolbox.rdm.calc import calc_rdm
from rsatoolbox.rdm.rdms import concat


data_dir = expanduser('~/data/rsatoolbox/mur32/derivatives/nilearn')

print('Loading..')
with open(join(data_dir, 'meta.json')) as fhandle:
    metadata = json.load(fhandle)
data = numpy.load(join(data_dir, 'data.npz'))
rois, betas, resids = data['rois'], data['betas'], data['resids']
subjects = sorted(set(metadata['subjects']))
conditions = metadata['conditions']
N_RUNS = 6
DOF = 38

rdm_list = []
for roi, region_name in enumerate(metadata['region_names']):
    for s, sub in enumerate(subjects):
        print(f'roi {region_name} sub {sub}')

        subject_runs = [r==sub for r in metadata['subjects']]
        roi_mask = rois[s, roi, :]
        patterns = betas[subject_runs][:, :, roi_mask]
        ds = Dataset(
            measurements=patterns.reshape(-1, roi_mask.sum()),
            descriptors=dict(sub=sub, roi=region_name),
            obs_descriptors=dict(
                run=numpy.repeat(numpy.arange(N_RUNS), len(conditions)),
                condition=numpy.tile(conditions, N_RUNS)
            )
        )

        runwise_prec_matrix = []
        for r in numpy.where(subject_runs)[0]:
            runwise_prec_matrix.append(
                prec_from_residuals(
                    resids[r, :, roi_mask].T,
                    dof=DOF,
                    method='shrinkage_diag'
                )
            )

        rdm_list.append(
            calc_rdm(
                dataset=ds,
                noise=runwise_prec_matrix,
                method='crossnobis',
                descriptor='condition',
                cv_descriptor='run',
            )
        )
data_rdms = concat(rdm_list)
del resids
del betas
