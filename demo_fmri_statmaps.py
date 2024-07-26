"""Script version of statmap-based fmri demo

This approach: store full statmaps
Other approach: store only masked roi patterns

"""
from os.path import expanduser, join
import json, warnings
import numpy
from rsatoolbox.data.dataset import Dataset
from rsatoolbox.data.noise import prec_from_residuals
from rsatoolbox.rdm.calc import calc_rdm
from rsatoolbox.rdm.rdms import concat
from rsatoolbox.vis.rdm_plot import show_rdm
import matplotlib.pyplot as plt


data_dir = expanduser('~/data/rsatoolbox/mur32/derivatives/nilearn')

print('Loading..')
with open(join(data_dir, 'meta.json')) as fhandle:
    metadata = json.load(fhandle)
data = numpy.load(join(data_dir, 'data.npz'))


subjects = metadata['subjects']
conditions = metadata['conditions']
dof = metadata['degrees_of_freedom']
N_RUNS = 6


rdm_list = []
for roi, region_name in enumerate(metadata['region_names']):
    for s, sub in enumerate(subjects):
        print(f'roi {region_name} sub {sub}')

        betas = data[f'betas_sub-{sub}_{region_name}']
        patterns = betas.reshape(-1, betas.shape[-1])
        ds = Dataset(
            measurements=patterns,
            descriptors=dict(sub=sub, roi=region_name),
            obs_descriptors=dict(
                run=numpy.repeat(numpy.arange(N_RUNS), len(conditions)),
                condition=numpy.tile(conditions, N_RUNS)
            )
        )

        runwise_prec_matrix = []
        resids = data[f'resids_sub-{sub}_{region_name}']
        for r in range(N_RUNS):
            runwise_prec_matrix.append(
                prec_from_residuals(
                    resids[r, :, :],
                    dof=dof,
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
        del rdm_list[-1].descriptors['noise']
data_rdms = concat(rdm_list)

for region_name in metadata['region_names']:
    roi_rdms = data_rdms.subset('roi', region_name)
    fig, _, _ = show_rdm(roi_rdms, show_colorbar='panel')
    plt.savefig(f'fmri_data/rdms_{region_name}.png')
    plt.close('all')